# stdlib
import time
from dataclasses import field
from functools import partial
from typing import Any, Dict, Tuple

# third party
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from jax.random import KeyArray
from pydantic import BaseModel, validator
from sklearn.model_selection import train_test_split

# synthcity absolute
import synthcity.logger as log


class NetworkConfig(BaseModel):
    task_type: str
    input_shape: int
    output_shape: int
    model_type: type
    hidden_layers: Tuple = (100,)
    batch_size: int = 128
    epochs: int = 5000
    min_epochs: int = 100
    print_epochs: int = 50
    learning_rate: float = 1e-3
    dropout: float = 0
    batchnorm: bool = False
    nonlin: str = "relu"
    patience: int = 10
    seed: int = 0
    optimizer: str = "adam"

    @validator("task_type", always=True)
    def _validate_task_type(cls: Any, v: str, values: Dict) -> str:
        if v not in ["regression", "classification"]:
            raise ValueError(f"Invalid task type {v}")

        return v

    @validator("nonlin", always=True)
    def _validate_nonlin(cls: Any, v: str, values: Dict) -> str:
        if v not in ["relu", "elu", "leaky_relu"]:
            raise ValueError(f"invalid nonlin optionan {v}")
        return v

    @validator("optimizer", always=True)
    def _validate_optimizer(cls: Any, v: str, values: Dict) -> str:
        if v not in ["sgd", "adam"]:
            raise ValueError(f"Invalid optimizer option {v}")
        return v

    @validator("dropout", always=True)
    def _validate_dropout(cls: Any, v: float, values: Dict) -> float:
        if v > 1:
            raise ValueError(f"Invalid dropout optional {v}. It must be <= 1")
        return v

    def __hash__(self) -> Any:
        return hash((type(self),) + tuple(self.__dict__.values()))


class MLP:
    def __init__(
        self,
        task_type: str,
        **kwargs: Any,
    ) -> None:
        self.task_type = task_type
        self.net_kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLP":
        if self.task_type == "classification":
            output_shape = len(np.unique(y))
            y = jax.nn.one_hot(y, output_shape)
        else:
            output_shape = 1

        input_shape = X.shape[1]
        self.config = NetworkConfig(
            task_type=self.task_type,
            input_shape=input_shape,
            output_shape=output_shape,
            model_type=BasicNetwork,
            **self.net_kwargs,
        )

        self.model = _training_loop(self.config, (X, y))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.config.task_type == "classification":
            preds = self.predict_proba(X)
            result = jnp.argmax(preds, -1).squeeze()
        else:
            result = (
                self.config.model_type(self.config)
                .apply({"params": self.model}, X)
                .squeeze()
            )

        return jax.device_get(result)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.config.task_type != "classification":
            raise ValueError(
                f"Invalid task type for predict_proba {self.config.task_type}"
            )

        preds = self.config.model_type(self.config).apply({"params": self.model}, X)
        return jax.device_get(jax.nn.softmax(preds))

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        preds = self.predict(X)

        return _accuracy(self.config, y, preds)


class BasicNetwork(nn.Module):
    config: NetworkConfig
    nonlins: Dict = field(
        default_factory=lambda: {
            "relu": nn.relu,
            "elu": nn.elu,
            "leaky_relu": nn.leaky_relu,
        }
    )

    def setup(self) -> None:
        if self.config.nonlin not in self.nonlins:
            raise ValueError(
                f"Invalid nonlinearity {self.config.nonlin}. Available: {self.nonlins}"
            )

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        for idx, size in enumerate(self.config.hidden_layers):
            if idx > 0 and self.config.dropout > 0:
                x = nn.Dropout(self.config.dropout)(x, deterministic=deterministic)
            x = nn.Dense(size)(x)
            x = self.nonlins[self.config.nonlin](x)
            if self.config.batchnorm:
                x = nn.BatchNorm(use_running_average=not deterministic)(x)
        x = nn.Dense(self.config.output_shape)(x)

        return x


@partial(jax.jit, static_argnums=(0,))
def _apply(config: NetworkConfig, params: Dict, X: jnp.ndarray) -> jnp.ndarray:
    if config.task_type == "classification":
        preds = config.model_type(config).apply({"params": params}, X)
        return jax.nn.softmax(preds)
    else:
        return config.model_type(config).apply({"params": params}, X)


def _accuracy(config: NetworkConfig, y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """Accuracy evaluator.

    Args:
        config: Network configuration.
        y_true: Labels
        y_pred: Predictions.

    Returns:
        float: the accuracy score.
    """
    if config.task_type == "classification":
        return jnp.mean(jnp.argmax(y_pred, -1) == jnp.argmax(y_true, -1))
    else:
        return jnp.mean(jnp.inner(y_true - y_pred, y_true - y_pred) / 2.0)


@partial(jax.jit, static_argnums=(0,))
def _loss_fn(
    config: NetworkConfig,
    params: Dict,
    dataset: Tuple[jnp.ndarray, jnp.ndarray],
    rng: KeyArray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Loss evaluator.

    Args:
        config: Network configuration.
        params: Network weights and biases.
        dataset: Training dataset
        rng: PRNG key
    Returns:
        loss: float
        predictions: jnp.ndarray
    """
    X, y = dataset
    preds = config.model_type(config).apply(
        {"params": params},
        X,
        deterministic=False,
        rngs={"dropout": rng},
        # mutable=["batch_stats"],
    )
    if config.task_type == "classification":
        loss = jnp.mean(optax.softmax_cross_entropy(logits=preds, labels=y))
    else:
        loss = jnp.mean(jnp.inner(y - preds, y - preds) / 2.0)
    return loss, preds


def _train_init(config: NetworkConfig, rng: KeyArray) -> TrainState:
    """Creates initial `TrainState`.

    Args:
        config: Network configuration.
        rng: PRNG key
    Returns:
        state: A new TrainState
    """
    params = config.model_type(config).init(rng, jnp.ones([config.input_shape]))[
        "params"
    ]
    if config.optimizer == "adam":
        tx = optax.adam(config.learning_rate)
    elif config.optimizer == "sgd":
        tx = optax.sgd(config.learning_rate)

    return TrainState.create(
        apply_fn=config.model_type(config).apply, params=params, tx=tx
    )


@partial(jax.jit, static_argnums=(0,))
def _train_step(
    config: NetworkConfig,
    dataset: Tuple[jnp.ndarray, jnp.ndarray],
    permutations: jnp.ndarray,
    step: int,
    step_state: Tuple[TrainState, jnp.float64, jnp.float64, KeyArray],
) -> Tuple[TrainState, jnp.float64, jnp.float64, KeyArray]:
    """Computes gradients, loss and accuracy for a single batch.

    Args:
        config: Network configuration.
        dataset: Training dataset
        permutations: Batch permutations
        step: Current batch permutation index
        step_state: Tuple of the current train state, accuracy, loss and the PRNG key.
    Returns:
        state: New TrainState
        loss: batch loss
        accuracy: batch accuracy
    """
    (state, accuracy_accumulator, loss_accumulator, rng) = step_state
    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True, argnums=1)

    rng, input_rng = jax.random.split(rng)

    X, y = dataset
    X_batch = X[permutations[step], ...]
    y_batch = y[permutations[step], ...]

    (loss, logits), grads = grad_fn(config, state.params, (X_batch, y_batch), input_rng)
    state = state.apply_gradients(grads=grads)

    accuracy_accumulator += _accuracy(config, y_batch, logits)
    loss_accumulator += loss

    return state, accuracy_accumulator, loss_accumulator, rng


@partial(jax.jit, static_argnums=(0,))
def _train_epoch(
    config: NetworkConfig,
    state: TrainState,
    train_ds: Tuple[jnp.ndarray, jnp.ndarray],
    rng: KeyArray,
) -> Tuple[TrainState, float, float, KeyArray]:
    """Train for a single epoch.
    Args:
        config: Network configuration.
        state: the current train state
        dataset: Training dataset
        rng: PRNG key
    Returns:
        state: New TrainState
        loss: epoch loss
        accuracy: epoch accuracy
        rng: new PRNG key
    """
    X, y = train_ds

    rng, input_rng = jax.random.split(rng)

    steps_per_epoch = len(X) // config.batch_size

    perms = jax.random.permutation(input_rng, len(X))
    perms = perms[: steps_per_epoch * config.batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, config.batch_size))

    step_state = (state, 0, 0, rng)
    partial_training_step = partial(_train_step, config, train_ds, perms)

    (state, accumulated_accuracy, accumulated_loss, rng) = jax.lax.fori_loop(
        0, steps_per_epoch, partial_training_step, step_state
    )

    train_accuracy = accumulated_accuracy / steps_per_epoch
    train_loss = accumulated_loss / steps_per_epoch

    return state, train_loss, train_accuracy, rng


def _training_loop(config: NetworkConfig, dataset: tuple) -> TrainState:
    """Train the model
    Args:
        config: Network configuration.
        dataset: Training dataset
    Returns:
        state: the final TrainState(params + optimizer)
    """
    # Prepare train/test datasets
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Prepare PRNG key
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)

    # Create initial TrainState
    state = _train_init(config, init_rng)

    # Train
    for epoch in range(config.epochs):
        start = time.time()
        state, train_loss, train_accuracy, rng = _train_epoch(
            config,
            state,
            (X_train, y_train),
            rng,
        )
        if epoch % config.print_epochs == 1:
            test_preds = _apply(config, state.params, X_test)
            test_accuracy = _accuracy(config, y_test, test_preds)
            log.info(
                f"[Epoch {epoch}] train/test loss: {train_loss}. train/test Acc: {train_accuracy} / {test_accuracy}. Duration: {time.time() - start}",
            )

    return state.params
