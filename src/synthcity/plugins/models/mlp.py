# stdlib
import time
from dataclasses import field
from functools import partial
from typing import Any, Dict, Iterable, Tuple

# third party
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from jax.random import KeyArray
from pydantic import BaseModel, validator
from sklearn.model_selection import train_test_split


class NetworkConfig(BaseModel):
    task_type: str
    model_type: type
    input_shape: int
    output_shape: int
    hidden_layers: Iterable[int]
    batch_size: int = 64
    epochs: int = 500
    min_epochs: int = 100
    learning_rate: float = 1e-3
    dropout: float = 0
    batchnorm: bool = True
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
                x = nn.BatchNorm(use_running_average=True)(x)
        return nn.Dense(self.config.output_shape)(x)


def accuracy(
    config: NetworkConfig, params: Dict, y_true: jnp.ndarray, y_pred: jnp.ndarray
) -> float:
    if config.task_type == "classification":
        return jnp.mean(jnp.argmax(y_pred, -1) == jnp.argmax(y_true, -1))
    else:
        return jnp.mean(jnp.inner(y_true - y_pred, y_true - y_pred) / 2.0)


@partial(jax.jit, static_argnums=(0,))
def loss_fn(
    config: NetworkConfig,
    params: Dict,
    dataset: Tuple[jnp.ndarray, jnp.ndarray],
    rng: KeyArray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    X, y = dataset
    preds = config.model_type(config).apply(
        {"params": params}, X, deterministic=False, rngs={"dropout": rng}
    )
    if config.task_type == "classification":
        loss = jnp.mean(optax.softmax_cross_entropy(logits=preds, labels=y))
    else:
        loss = jnp.mean(jnp.inner(y - preds, y - preds) / 2.0)
    return loss, preds


def create_train_state(config: NetworkConfig, rng: KeyArray) -> TrainState:
    """Creates initial `TrainState`."""
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
def train_step(
    config: NetworkConfig,
    dataset: Tuple[jnp.ndarray, jnp.ndarray],
    permutations: jnp.ndarray,
    step: int,
    step_state: Tuple[TrainState, jnp.float64, jnp.float64],
    rng: KeyArray,
) -> Tuple[TrainState, jnp.float64, jnp.float64]:
    """Computes gradients, loss and accuracy for a single batch."""
    (state, accuracy_accumulator, loss_accumulator) = step_state
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True, argnums=1)

    X, y = dataset
    X_batch = X[permutations[step], ...]
    y_batch = y[permutations[step], ...]

    (loss, logits), grads = grad_fn(config, state.params, (X_batch, y_batch), rng)
    state = state.apply_gradients(grads=grads)

    accuracy_accumulator += accuracy(config, state, y_batch, logits)
    loss_accumulator += loss

    return state, accuracy_accumulator, loss_accumulator


@partial(jax.jit, static_argnums=(0,))
def train_epoch(
    config: NetworkConfig,
    state: TrainState,
    train_ds: Tuple[jnp.ndarray, jnp.ndarray],
    rng: KeyArray,
) -> Tuple[TrainState, float, float, KeyArray]:
    """Train for a single epoch."""
    X, y = train_ds

    rng, input_rng = jax.random.split(rng)

    steps_per_epoch = len(X) // config.batch_size

    perms = jax.random.permutation(input_rng, len(X))
    perms = perms[: steps_per_epoch * config.batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, config.batch_size))

    step_state = (state, 0, 0)
    partial_training_step = partial(train_step, config, train_ds, perms)

    (state, accumulated_accuracy, accumulated_loss) = jax.lax.fori_loop(
        0, steps_per_epoch, partial_training_step, step_state, rng
    )

    train_accuracy = accumulated_accuracy / steps_per_epoch
    train_loss = accumulated_loss / steps_per_epoch

    return state, train_loss, train_accuracy, rng


def train_and_evaluate(config: NetworkConfig, dataset: tuple) -> TrainState:
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print(f"X = {X_train[0].shape} y = {y_train[1].shape}")
    rng = jax.random.PRNGKey(config.seed)

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(config, init_rng)

    for epoch in range(config.epochs):
        start = time.time()
        state, train_loss, train_accuracy, rng = train_epoch(
            config,
            state,
            (X_train, y_train),
            rng,
        )
        if epoch % 10 == 1:
            test_preds = config.model_type(config).apply(
                {"params": state.params}, X_test
            )
            test_accuracy = accuracy(config, state, y_test, test_preds)
            print(
                f"[Epoch {epoch}] train/test loss: {train_loss}. train/test Acc: {train_accuracy} / {test_accuracy}. Duration: {time.time() - start}",
                flush=True,
            )

    return state
