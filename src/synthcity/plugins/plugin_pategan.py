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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# synthcity absolute
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.models import GAN


class Teachers:
    def __init__(
        self,
        n_teachers: int,
        partition_size: int,
        lamda: float = 1,  # PATE noise size
        template: str = "linear",
    ) -> None:
        self.n_teachers = n_teachers
        self.partition_size = partition_size
        self.lamda = lamda
        if template == "xgboost":
            self.model_template = XGBClassifier
        else:
            self.model_template = LogisticRegression

    def fit(self, X: np.ndarray, generator: Any) -> "Teachers":
        # 1. train teacher models
        self.teacher_models: list = []

        permutations = np.random.permutation(len(X))

        for tidx in range(self.n_teachers):
            teacher_idx = permutations[
                int(tidx * self.partition_size) : int((tidx + 1) * self.partition_size)
            ]
            teacher_X = X[teacher_idx, :]

            g_mb = generator(self.partition_size)

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

            model = self.model_template()
            model.fit(x_comb, y_comb)

            self.teacher_models.append(model)

        return self

    def pate_lamda(self, x: np.ndarray) -> Tuple[int, int, int]:
        """Returns PATE_lambda(x).

        Args:
          - x: feature vector

        Returns:
          - n0, n1: the number of label 0 and 1, respectively
          - out: label after adding laplace noise.
        """

        y_hat: List = []

        for teacher in self.teacher_models:
            temp_y = teacher.predict(np.reshape(x, [1, -1]))
            y_hat = y_hat + [temp_y]

        y_hat_np = np.asarray(y_hat, dtype=int)
        n0 = sum(y_hat_np == 0)
        n1 = sum(y_hat_np == 1)

        lap_noise = np.random.laplace(loc=0.0, scale=self.lamda)

        out = (n1 + lap_noise) / float(n0 + n1)
        out = int(out > 0.5)

        return n0, n1, out


class PATEGAN:
    """Basic PATE-GAN framework."""

    def __init__(
        self,
        # GAN
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 100,
        generator_nonlin: str = "tanh",
        generator_n_iter: int = 100,
        generator_dropout: float = 0,
        discriminator_n_layers_hidden: int = 2,
        discriminator_n_units_hidden: int = 100,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        batch_size: int = 64,
        seed: int = 0,
        clipping_value: int = 1,
        # Privacy
        n_teachers: int = 10,
        teacher_template: str = "linear",
        epsilon: float = 1.0,
        delta: float = 0.00001,
        lamda: float = 1,
        alpha: int = 20,
    ) -> None:
        self.encoder = MinMaxScaler()
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
        self.seed = seed
        self.clipping_value = clipping_value
        # Privacy
        self.n_teachers = n_teachers
        self.teacher_template = teacher_template
        self.epsilon = epsilon
        self.delta = delta
        self.lamda = lamda
        self.alpha = alpha

    def fit(
        self,
        X_train: np.ndarray,
    ) -> "PATEGAN":
        X_train = self.encoder.fit_transform(X_train)

        X_train = torch.from_numpy(np.asarray(X_train))
        features = X_train.shape[1]

        self.model = GAN(
            n_features=features,
            n_units_latent=features,
            batch_size=self.batch_size,
            generator_n_layers_hidden=self.generator_n_layers_hidden,
            generator_n_units_hidden=self.generator_n_units_hidden,
            generator_nonlin=self.generator_nonlin,
            generator_nonlin_out=[("tanh", features)],
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
        )
        partition_data_no = int(len(X_train) / self.n_teachers)

        loader = self.model.dataloader(X_train)

        # alpha initialize
        self.alpha_dict = np.zeros([self.alpha])

        # initialize epsilon_hat
        epsilon_hat = 0

        # Iterations
        while epsilon_hat < self.epsilon:
            # 1. Train teacher models
            teachers = Teachers(
                self.n_teachers,
                partition_data_no,
                lamda=self.lamda,
                template=self.teacher_template,
            )
            teachers.fit(X_train.cpu().numpy(), self.model.generate)

            # 2. Student training
            def fake_labels_generator(X: torch.Tensor) -> torch.Tensor:
                Y_mb: list = []
                for j in range(len(X)):
                    n0, n1, r_j = teachers.pate_lamda(X[j, :].detach().cpu())
                    Y_mb = Y_mb + [r_j]

                    # Update moments accountant
                    q = self._update_moments_accountant(n0, n1)

                    # Compute alpha
                    self._update_alpha(q)

                # PATE labels for X
                return torch.from_numpy(
                    np.reshape(np.asarray(Y_mb, dtype=int), [-1, 1])
                )

            self.model.train_epoch(loader, fake_labels_generator=fake_labels_generator)

            # epsilon_hat computation
            curr_list: List = []
            for lidx in range(self.alpha):
                temp_alpha = (self.alpha_dict[lidx] + np.log(1 / self.delta)) / float(
                    lidx + 1
                )
                curr_list = curr_list + [temp_alpha]

            epsilon_hat = np.min(curr_list)

        return self

    def _update_moments_accountant(self, n0: int, n1: int) -> float:
        # Update moments accountant
        q = (
            np.log(2 + self.lamda * abs(n0 - n1))
            - np.log(4.0)
            - (self.lamda * abs(n0 - n1))
        )
        return np.exp(q)

    def _update_alpha(self, q: float) -> Dict:
        # Compute alpha
        for lidx in range(self.alpha):
            temp1 = 2 * (self.lamda**2) * (lidx + 1) * (lidx + 2)
            temp2 = (1 - q) * (
                ((1 - q) / (1 - q * np.exp(2 * self.lamda) + 1e-8)) ** (lidx + 1)
            ) + q * np.exp(2 * self.lamda * (lidx + 1))
            self.alpha_dict[lidx] += np.min([temp1, np.log(temp2)])
        return self.alpha_dict

    def sample(self, count: int) -> np.ndarray:
        with torch.no_grad():
            return self.encoder.inverse_transform(self.model.generate(count))


class PATEGANPlugin(Plugin):
    """PATEGAN plugin.

    Args:
        generator_n_layers_hidden: int
            Number of hidden layers in the generator
        generator_n_units_hidden: int
            Number of hidden units in each layer of the Generator
        generator_nonlin: string, default 'tanh'
            Nonlinearity to use in the generator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        generator_n_iter: int
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
        seed: int
            Seed used
        clipping_value: int, default 1
            Gradients clipping value
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
    Example:
        >>> from synthcity.plugins import Plugins
        >>> plugin = Plugins().get("pategan")
        >>> from sklearn.datasets import load_iris
        >>> X = load_iris()
        >>> plugin.fit(X)
        >>> plugin.generate()
    """

    def __init__(
        self,
        # GAN
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 100,
        generator_nonlin: str = "tanh",
        generator_n_iter: int = 100,
        generator_dropout: float = 0,
        discriminator_n_layers_hidden: int = 2,
        discriminator_n_units_hidden: int = 100,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        batch_size: int = 64,
        seed: int = 0,
        clipping_value: int = 1,
        # Privacy
        n_teachers: int = 10,
        teacher_template: str = "linear",
        epsilon: float = 1.0,
        delta: float = 0.00001,
        lamda: float = 1,
        alpha: int = 20,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.model = PATEGAN(
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
            seed=seed,
            clipping_value=clipping_value,
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
        return "gan"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="generator_n_layers_hidden", low=1, high=5),
            IntegerDistribution(
                name="generator_n_units_hidden", low=50, high=500, step=50
            ),
            CategoricalDistribution(
                name="generator_nonlin", choices=["relu", "leaky_relu", "tanh", "elu"]
            ),
            IntegerDistribution(name="generator_n_iter", low=100, high=500, step=100),
            FloatDistribution(name="generator_dropout", low=0, high=0.2),
            IntegerDistribution(name="discriminator_n_layers_hidden", low=1, high=5),
            IntegerDistribution(
                name="discriminator_n_units_hidden", low=50, high=500, step=50
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
            IntegerDistribution(name="n_teachers", low=2, high=15),
            CategoricalDistribution(
                name="teacher_template", choices=["linear", "xgboost"]
            ),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "PATEGANPlugin":
        self.model.fit(X)

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.sample, count, syn_schema)


plugin = PATEGANPlugin
