"""PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees Codebase.

Reference: James Jordon, Jinsung Yoon, Mihaela van der Schaar,
"PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees,"
International Conference on Learning Representations (ICLR), 2019.
Paper link: https://openreview.net/forum?id=S1zk9iRqF7
Last updated Date: Feburuary 15th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
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

# synthcity absolute
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.models import GAN


class Teachers:
    def __init__(
        self,
        n_teachers: int,
        partition_size: int,
        lamda: float = 1,  # PATE noise size
        model_template: Any = LogisticRegression,
    ) -> None:
        self.n_teachers = n_teachers
        self.partition_size = partition_size
        self.model_template = model_template
        self.lamda = lamda

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
          - teacher_models: a list of teacher models

        Returns:
          - n0, n1: the number of label 0 and 1, respectively
          - out: label after adding laplace noise.
        """

        y_hat: List = []

        for teacher in self.teacher_models:
            temp_y = teacher.predict(np.reshape(x, [1, -1]))
            y_hat = y_hat + [temp_y]

        y_hat_np = np.asarray(y_hat)
        n0 = sum(y_hat_np == 0)
        n1 = sum(y_hat_np == 1)

        lap_noise = np.random.laplace(loc=0.0, scale=self.lamda)

        out = (n1 + lap_noise) / float(n0 + n1)
        out = int(out > 0.5)

        return n0, n1, out


class PATEGAN:
    """Basic PATE-GAN framework.
    Args:
        - epochs: the number of student training iterations
        - batch_size: the number of batch size for training student and generator
        - k: the number of teachers
        - epsilon, delta: Differential privacy parameters
        - lamda: noise size
    """

    def __init__(
        self,
        epochs: int = 100,  # the number of student training iterations
        discr_epochs: int = 1,  # the number of student training iterations
        batch_size: int = 64,  # the number of batch size for training student and generator
        n_teachers: int = 10,  # the number of teachers
        epsilon: float = 1.0,  # Differential privacy parameters (epsilon)
        delta: float = 0.00001,  # Differential privacy parameters (delta)
        lamda: float = 1,  # PATE noise size
        learning_rate: float = 1e-4,
        alpha: int = 20,
        clipping_value: float = 0.01,
    ) -> None:
        self.epochs = epochs
        self.discr_epochs = discr_epochs
        self.batch_size = batch_size
        self.n_teachers = n_teachers
        self.epsilon = epsilon
        self.delta = delta
        self.lamda = lamda
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.clipping_value = clipping_value

    def fit(
        self,
        X_train: np.ndarray,
    ) -> "PATEGAN":
        """Basic PATE-GAN framework.

        Args:
            - x_train: training data
            - parameters: PATE-GAN parameters
            - epochs: the number of student training iterations
            - batch_size: the number of batch size for training student and generator
            - k: the number of teachers
            - epsilon, delta: Differential privacy parameters
            - lamda: noise size

        Returns:
          - x_train_hat: generated training data by differentially private generator
        """

        self.encoder = MinMaxScaler().fit(np.asarray(X_train))
        X_train = torch.from_numpy(np.asarray(X_train))
        features = X_train.shape[1]

        self.model = GAN(
            n_features=features,
            n_units_latent=features,
            batch_size=self.batch_size,
            generator_n_layers_hidden=2,
            generator_n_units_hidden=4 * features,
            generator_nonlin="tanh",
            generator_nonlin_out="sigmoid",
            generator_lr=self.learning_rate,
            generator_residual=True,
            generator_n_iter=self.epochs,
            generator_batch_norm=False,
            generator_dropout=0,
            generator_weight_decay=1e-3,
            discriminator_n_units_hidden=4 * features,
            discriminator_n_iter=self.discr_epochs,
            discriminator_nonlin="leaky_relu",
            discriminator_batch_norm=False,
            discriminator_dropout=0.1,
            discriminator_lr=self.learning_rate,
            discriminator_weight_decay=1e-3,
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
            teachers = Teachers(self.n_teachers, partition_data_no, lamda=self.lamda)
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
                return torch.from_numpy(np.reshape(np.asarray(Y_mb), [-1, 1]))

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
                ((1 - q) / (1 - q * np.exp(2 * self.lamda))) ** (lidx + 1)
            ) + q * np.exp(2 * self.lamda * (lidx + 1))
            self.alpha_dict[lidx] += np.min([temp1, np.log(temp2)])
        return self.alpha_dict

    def sample(self, count: int) -> np.ndarray:
        with torch.no_grad():
            x_hat = self.model.generate(count)
            return self.encoder.inverse_transform(x_hat)


class PATEGANPlugin(Plugin):
    """PATEGAN plugin.

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
        epochs: int = 100,
        discr_epochs: int = 1,
        batch_size: int = 64,
        n_teachers: int = 10,
        epsilon: float = 1.0,
        delta: float = 0.00001,
        lamda: float = 1,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.model = PATEGAN(
            epochs=epochs,
            discr_epochs=discr_epochs,
            batch_size=batch_size,
            n_teachers=n_teachers,
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
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "PATEGANPlugin":
        self.model.fit(X)

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.sample, count, syn_schema)


plugin = PATEGANPlugin
