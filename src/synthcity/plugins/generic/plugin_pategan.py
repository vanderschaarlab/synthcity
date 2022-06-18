"""PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees Codebase.

Reference: James Jordon, Jinsung Yoon, Mihaela van der Schaar,
"PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees,"
International Conference on Learning Representations (ICLR), 2019.
Paper link: https://openreview.net/forum?id=S1zk9iRqF7
"""
# stdlib
from typing import Any, Dict, List, Tuple

# third party
import numpy as np
import pandas as pd

# Necessary packages
import torch

# Necessary packages
from pydantic import validate_arguments
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# synthcity absolute
import synthcity.logger as log
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.models import TabularGAN
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.core.serializable import Serializable


class Teachers(Serializable):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_teachers: int,
        partition_size: int,
        lamda: float = 1,  # PATE noise size
        template: str = "xgboost",
    ) -> None:
        super().__init__()

        self.n_teachers = n_teachers
        self.partition_size = partition_size
        self.lamda = lamda
        self.model_args: dict = {}
        if template == "xgboost":
            self.model_template = XGBClassifier
            self.model_args = {
                "verbosity": 0,
                "depth": 4,
            }
        else:
            self.model_template = LogisticRegression
            self.model_args = {
                "solver": "sag",
                "max_iter": 10000,
            }

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: np.ndarray, generator: Any) -> Any:
        # 1. train teacher models
        self.teacher_models: list = []

        permutations = np.random.permutation(len(X))

        log.debug("Training teachers")
        for tidx in range(self.n_teachers):
            log.debug(f"  >> Training teacher {tidx}")
            teacher_idx = permutations[
                int(tidx * self.partition_size) : int((tidx + 1) * self.partition_size)
            ]
            teacher_X = X[teacher_idx, :]

            g_mb = generator(self.partition_size).detach().cpu()

            idx = np.random.permutation(len(teacher_X[:, 0]))
            x_mb = teacher_X[idx[: self.partition_size], :]

            x_comb = np.concatenate((x_mb, g_mb), axis=0)
            y_comb = np.concatenate(
                (
                    np.ones(
                        [
                            self.partition_size,
                        ]
                    ),
                    np.zeros(
                        [
                            self.partition_size,
                        ]
                    ),
                ),
                axis=0,
            )

            model = self.model_template(**self.model_args)
            model.fit(x_comb, y_comb)

            self.teacher_models.append(model)

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def pate_lamda(self, x: np.ndarray) -> Tuple[int, int, int]:
        """Returns PATE_lambda(x).

        Args:
          - x: feature vector

        Returns:
          - n0, n1: the number of label 0 and 1, respectively
          - out: label after adding laplace noise.
        """

        predictions = []

        for tidx, teacher in enumerate(self.teacher_models):
            y_pred = teacher.predict(x)
            predictions.append(y_pred)

        y_hat = np.vstack(predictions)  # (n_teachers, batch_size)

        n0 = np.sum(y_hat == 0, axis=0)  # (batch_size, )
        n1 = np.sum(y_hat == 1, axis=0)  # (batch_size, )

        lap_noise = np.random.laplace(loc=0.0, scale=self.lamda, size=n0.shape)

        out = (n1 + lap_noise) / (n0 + n1 + 1e-8)  # (batch_size, )
        out = (out > 0.5).astype(int)

        return n0, n1, out


class PATEGAN(Serializable):
    """Basic PATE-GAN framework."""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        # GAN
        max_iter: int = 10,
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 100,
        generator_nonlin: str = "tanh",
        generator_n_iter: int = 5,
        generator_dropout: float = 0,
        discriminator_n_layers_hidden: int = 2,
        discriminator_n_units_hidden: int = 100,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        batch_size: int = 500,
        random_state: int = 0,
        clipping_value: int = 0,
        encoder_max_clusters: int = 20,
        # Privacy
        n_teachers: int = 10,
        teacher_template: str = "linear",
        epsilon: float = 1.0,
        delta: float = 0.00001,
        lamda: float = 1,
        alpha: int = 20,
        encoder: Any = None,
    ) -> None:
        super().__init__()

        self.max_iter = max_iter
        self.generator_n_layers_hidden = generator_n_layers_hidden
        self.generator_n_units_hidden = generator_n_units_hidden
        self.generator_nonlin = generator_nonlin
        self.generator_n_iter = generator_n_iter
        self.generator_dropout = generator_dropout
        self.discriminator_n_layers_hidden = discriminator_n_layers_hidden
        self.discriminator_n_units_hidden = discriminator_n_units_hidden
        self.discriminator_nonlin = discriminator_nonlin
        self.discriminator_n_iter = discriminator_n_iter
        self.discriminator_dropout = discriminator_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.random_state = random_state
        self.clipping_value = clipping_value
        # Privacy
        self.n_teachers = n_teachers
        self.teacher_template = teacher_template
        self.epsilon = epsilon
        self.delta = delta
        self.lamda = lamda
        self.alpha = alpha
        self.encoder_max_clusters = encoder_max_clusters
        self.encoder = encoder

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        X_train: pd.DataFrame,
    ) -> Any:
        self.columns = X_train.columns

        features = X_train.shape[1]

        self.model = TabularGAN(
            X_train,
            n_units_latent=features,
            batch_size=self.batch_size,
            generator_n_layers_hidden=self.generator_n_layers_hidden,
            generator_n_units_hidden=self.generator_n_units_hidden,
            generator_nonlin=self.generator_nonlin,
            generator_nonlin_out_discrete="softmax",
            generator_nonlin_out_continuous="tanh",
            generator_lr=self.lr,
            generator_residual=True,
            generator_n_iter=self.generator_n_iter,
            generator_batch_norm=False,
            generator_dropout=0,
            generator_weight_decay=self.weight_decay,
            discriminator_n_units_hidden=self.discriminator_n_units_hidden,
            discriminator_n_layers_hidden=self.discriminator_n_layers_hidden,
            discriminator_n_iter=self.discriminator_n_iter,
            discriminator_nonlin=self.discriminator_nonlin,
            discriminator_batch_norm=False,
            discriminator_dropout=self.discriminator_dropout,
            discriminator_lr=self.lr,
            discriminator_weight_decay=self.weight_decay,
            clipping_value=self.clipping_value,
            encoder_max_clusters=self.encoder_max_clusters,
            encoder=self.encoder,
            n_iter_print=2,
        )
        X_train_enc = self.model.encode(X_train)

        partition_data_no = len(X_train_enc)
        if self.n_teachers > 0:
            partition_data_no = int(len(X_train_enc) / self.n_teachers)

        # alpha initialize
        self.alpha_dict = np.zeros([self.alpha])

        # initialize epsilon_hat
        epsilon_hat = 0

        # Iterations
        it = 0
        while epsilon_hat < self.epsilon and it < self.max_iter:
            it += 1
            log.debug(
                f"[pategan it {it}] epsilon_hat = {epsilon_hat}. self.epsilon = {self.epsilon}"
            )

            log.debug(f"[pategan it {it}] 1. Train teacher models")

            # 1. Train teacher models
            teachers = Teachers(
                self.n_teachers,
                partition_data_no,
                lamda=self.lamda,
                template=self.teacher_template,
            )
            teachers.fit(np.asarray(X_train_enc), self.model)

            log.debug(f"[pategan it {it}] 2. GAN training")

            # 2. Student training
            def fake_labels_generator(X: torch.Tensor) -> torch.Tensor:
                if self.n_teachers == 0:
                    return torch.zeros((len(X),))

                X_batch = pd.DataFrame(X.detach().cpu().numpy())

                n0_mb, n1_mb, Y_mb = teachers.pate_lamda(np.asarray(X_batch))
                if np.sum(Y_mb) >= len(X) / 2:
                    log.debug(
                        f"[pategan it {it}] Teachers high error-rate: n0 = {len(X) - np.sum(Y_mb)}, n1 = {np.sum(Y_mb)}"
                    )
                    return torch.zeros((len(X),))

                for j in range(len(X_batch)):
                    n0, n1 = n0_mb[j], n1_mb[j]
                    # Update moments accountant
                    q = self._update_moments_accountant(n0, n1)

                    # Compute alpha
                    self._update_alpha(q)

                # PATE labels for X
                return torch.from_numpy(
                    np.reshape(np.asarray(Y_mb, dtype=int), [-1, 1])
                )

            self.model.fit(
                X_train_enc, fake_labels_generator=fake_labels_generator, encoded=True
            )

            # epsilon_hat computation
            curr_list: List = []
            for lidx in range(self.alpha):
                temp_alpha = (self.alpha_dict[lidx] + np.log(1 / self.delta)) / float(
                    lidx + 1
                )
                curr_list = curr_list + [temp_alpha]

            epsilon_hat = np.min(curr_list)
            log.debug(f"[pategan it {it}] 3. eps update {epsilon_hat}")

        log.debug("pategan training done")
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _update_moments_accountant(self, n0: int, n1: int) -> float:
        # Update moments accountant
        q = (
            np.log(2 + self.lamda * abs(n0 - n1))
            - np.log(4.0)
            - (self.lamda * abs(n0 - n1))
        )
        return np.exp(q)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _update_alpha(self, q: float) -> Dict:
        # Compute alpha
        for lidx in range(self.alpha):
            temp1 = 2 * (self.lamda**2) * (lidx + 1) * (lidx + 2)
            temp2 = (1 - q) * (
                ((1 - q) / (1 - q * np.exp(2 * self.lamda) + 1e-8)) ** (lidx + 1)
            ) + q * np.exp(2 * self.lamda * (lidx + 1))
            self.alpha_dict[lidx] += np.min([temp1, np.log(temp2)])
        return self.alpha_dict

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def sample(self, count: int) -> np.ndarray:
        samples = self.model(count).detach().cpu().numpy()
        return self.model.decode(pd.DataFrame(samples))


class PATEGANPlugin(Plugin):
    """PATEGAN plugin.

    Args:
        generator_n_layers_hidden: int
            Number of hidden layers in the generator
        generator_n_units_hidden: int
            Number of hidden units in each layer of the Generator
        generator_nonlin: string, default 'tanh'
            Nonlinearity to use in the generator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        n_iter: int
            Maximum number of iterations in the Generator.
        generator_dropout: float
            Dropout value. If 0, the dropout is not used.
        discriminator_n_layers_hidden: int
            Number of hidden layers in the discriminator
        discriminator_n_units_hidden: int
            Number of hidden units in each layer of the discriminator
        discriminator_nonlin: string, default 'leaky_relu'
            Nonlinearity to use in the discriminator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        discriminator_n_iter: int
            Maximum number of iterations in the discriminator.
        discriminator_dropout: float
            Dropout value for the discriminator. If 0, the dropout is not used.
        lr: float
            learning rate for optimizer. step_size equivalent in the JAX version.
        weight_decay: float
            l2 (ridge) penalty for the weights.
        batch_size: int
            Batch size
        random_state: int
            random_state used
        clipping_value: int, default 0
            Gradients clipping value. Zero disables the feature
        n_teachers: int
            Number of teachers to train
        teacher_template: str
            Model to use for the teachers. Can be linear, xgboost.
        epsilon: float
            Differential privacy parameter
        delta: float
            Differential privacy parameter
        lambda: float
            Noise size
        encoder_max_clusters: int
            The max number of clusters to create for continuous columns when encoding


    Example:
        >>> from synthcity.plugins import Plugins
        >>> plugin = Plugins().get("pategan")
        >>> from sklearn.datasets import load_iris
        >>> X = load_iris()
        >>> plugin.fit(X)
        >>> plugin.generate()
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        # GAN
        n_iter: int = 10,
        generator_n_iter: int = 10,
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 100,
        generator_nonlin: str = "tanh",
        generator_dropout: float = 0,
        discriminator_n_layers_hidden: int = 2,
        discriminator_n_units_hidden: int = 100,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        batch_size: int = 500,
        random_state: int = 0,
        clipping_value: int = 0,
        encoder_max_clusters: int = 20,
        # Privacy
        n_teachers: int = 10,
        teacher_template: str = "xgboost",
        epsilon: float = 10.0,
        delta: float = 0.00001,
        lamda: float = 1,
        alpha: int = 20,
        encoder: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.model = PATEGAN(
            max_iter=n_iter,
            generator_n_layers_hidden=generator_n_layers_hidden,
            generator_n_units_hidden=generator_n_units_hidden,
            generator_nonlin=generator_nonlin,
            generator_n_iter=generator_n_iter,
            generator_dropout=generator_dropout,
            discriminator_n_layers_hidden=discriminator_n_layers_hidden,
            discriminator_n_units_hidden=discriminator_n_units_hidden,
            discriminator_nonlin=discriminator_nonlin,
            discriminator_n_iter=discriminator_n_iter,
            discriminator_dropout=discriminator_dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            random_state=random_state,
            clipping_value=clipping_value,
            encoder_max_clusters=encoder_max_clusters,
            encoder=encoder,
            # Privacy
            n_teachers=n_teachers,
            teacher_template=teacher_template,
            epsilon=epsilon,
            delta=delta,
            lamda=lamda,
        )

    @staticmethod
    def name() -> str:
        return "pategan"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="n_iter", low=1, high=15),
            IntegerDistribution(name="generator_n_layers_hidden", low=1, high=4),
            IntegerDistribution(
                name="generator_n_units_hidden", low=50, high=150, step=50
            ),
            CategoricalDistribution(
                name="generator_nonlin", choices=["relu", "leaky_relu", "tanh", "elu"]
            ),
            IntegerDistribution(name="generator_n_iter", low=1, high=10),
            FloatDistribution(name="generator_dropout", low=0, high=0.2),
            IntegerDistribution(name="discriminator_n_layers_hidden", low=1, high=4),
            IntegerDistribution(
                name="discriminator_n_units_hidden", low=50, high=150, step=50
            ),
            CategoricalDistribution(
                name="discriminator_nonlin",
                choices=["relu", "leaky_relu", "tanh", "elu"],
            ),
            IntegerDistribution(name="discriminator_n_iter", low=1, high=5),
            FloatDistribution(name="discriminator_dropout", low=0, high=0.2),
            CategoricalDistribution(name="lr", choices=[1e-3, 2e-4, 1e-4]),
            CategoricalDistribution(name="weight_decay", choices=[1e-3, 1e-4]),
            CategoricalDistribution(name="batch_size", choices=[64, 128, 256, 512]),
            IntegerDistribution(name="n_teachers", low=5, high=15),
            CategoricalDistribution(
                name="teacher_template", choices=["linear", "xgboost"]
            ),
            IntegerDistribution(name="encoder_max_clusters", low=2, high=20),
            FloatDistribution(name="lamda", low=1, high=10),
            CategoricalDistribution(
                name="delta", choices=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
            ),
            IntegerDistribution(name="alpha", low=2, high=50),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "PATEGANPlugin":
        self.model.fit(X.dataframe())

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.sample, count, syn_schema)


plugin = PATEGANPlugin
