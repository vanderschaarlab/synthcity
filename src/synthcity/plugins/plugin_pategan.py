"""PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees Codebase.

Reference: James Jordon, Jinsung Yoon, Mihaela van der Schaar,
"PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees,"
International Conference on Learning Representations (ICLR), 2019.
Paper link: https://openreview.net/forum?id=S1zk9iRqF7
Last updated Date: Feburuary 15th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
"""
# stdlib
from typing import Any, List, Tuple

# third party
import numpy as np
import pandas as pd

# Necessary packages
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from torch import nn

# synthcity absolute
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class Generator(nn.Module):
    def __init__(
        self,
        z_dim: int,
        generator_h_dim: int,
        dim: int,
    ) -> None:
        super(Generator, self).__init__()
        # Generator
        self.model = nn.Sequential(
            nn.Linear(z_dim, generator_h_dim),
            nn.Tanh(),
            nn.Linear(generator_h_dim, generator_h_dim),
            nn.Tanh(),
            nn.Linear(generator_h_dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class Student(nn.Module):
    def __init__(self, dim: int, student_h_dim: int) -> None:
        super(Student, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, student_h_dim),
            nn.ReLU(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)


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
        epochs: int = 1,  # the number of student training iterations
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
        x_train: np.ndarray,
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

        self.encoder = MinMaxScaler().fit(np.asarray(x_train))
        x_train = torch.from_numpy(np.asarray(x_train))

        # alpha initialize
        alpha = np.zeros([self.alpha])

        # initialize epsilon_hat
        epsilon_hat = 0

        # Network parameters
        no, dim = x_train.shape
        # Random sample dimensions
        self.z_dim = int(dim)

        # Student hidden dimension
        student_h_dim = int(dim)
        # Generator hidden dimension
        generator_h_dim = int(4 * dim)

        # Partitioning the data into k subsets
        x_partition: List = []
        partition_data_no = int(no / self.n_teachers)

        idx = np.random.permutation(no)

        for i in range(self.n_teachers):
            temp_idx = idx[
                int(i * partition_data_no) : int((i + 1) * partition_data_no)
            ]
            temp_x = x_train[temp_idx, :]
            x_partition = x_partition + [temp_x]

        # NN variables
        self.generator = Generator(self.z_dim, generator_h_dim, dim)
        self.student = Student(dim, student_h_dim)

        # Loss
        def student_loss(Y: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
            return torch.mean(Y * generated) - torch.mean((1 - Y) * generated)

        def generator_loss(generated: torch.Tensor) -> torch.Tensor:
            return -torch.mean(generated)

        # Optimizer
        S_solver = torch.optim.RMSprop(self.student.parameters(), lr=self.learning_rate)
        G_solver = torch.optim.RMSprop(
            self.generator.parameters(), lr=self.learning_rate
        )

        S_solver.zero_grad()
        G_solver.zero_grad()

        # Iterations
        while epsilon_hat < self.epsilon:

            # 1. Train teacher models
            teacher_models: List = []

            for _ in range(self.n_teachers):

                Z_mb = self.sample_Z(partition_data_no, self.z_dim)
                G_mb = self.generator(Z_mb).detach().cpu().numpy()

                temp_x = x_partition[i]
                idx = np.random.permutation(len(temp_x[:, 0]))
                X_mb = temp_x[idx[:partition_data_no], :]

                X_comb = np.concatenate((X_mb, G_mb), axis=0)
                Y_comb = np.concatenate(
                    (
                        np.ones(
                            [
                                partition_data_no,
                            ]
                        ),
                        np.zeros(
                            [
                                partition_data_no,
                            ]
                        ),
                    ),
                    axis=0,
                )

                model = LogisticRegression()
                model.fit(X_comb, Y_comb)
                teacher_models = teacher_models + [model]

            # 2. Student training
            for _ in range(self.epochs):

                Z_mb = self.sample_Z(self.batch_size, self.z_dim)
                G_mb = self.generator(Z_mb).detach().cpu().numpy()
                Y_mb: List = []

                for j in range(self.batch_size):
                    n0, n1, r_j = self._pate_lamda(G_mb[j, :], teacher_models)
                    Y_mb = Y_mb + [r_j]

                    # Update moments accountant
                    q = (
                        np.log(2 + self.lamda * abs(n0 - n1))
                        - np.log(4.0)
                        - (self.lamda * abs(n0 - n1))
                    )
                    q = np.exp(q)

                    # Compute alpha
                    for lidx in range(self.alpha):
                        temp1 = 2 * (self.lamda**2) * (lidx + 1) * (lidx + 2)
                        temp2 = (1 - q) * (
                            ((1 - q) / (1 - q * np.exp(2 * self.lamda))) ** (lidx + 1)
                        ) + q * np.exp(2 * self.lamda * (lidx + 1))
                        alpha[lidx] = alpha[lidx] + np.min([temp1, np.log(temp2)])

                # PATE labels for G_mb
                Y_mb = torch.from_numpy(np.reshape(np.asarray(Y_mb), [-1, 1]))

                # Update student
                G_sample = self.generator(Z_mb)
                S_fake = self.student(G_sample)

                loss = student_loss(Y_mb, S_fake)
                loss.backward()

                nn.utils.clip_grad_norm_(self.student.parameters(), self.clipping_value)

                S_solver.step()

            # Generator Update
            Z_mb = self.sample_Z(self.batch_size, self.z_dim)
            G_sample_mb = self.generator(Z_mb)
            S_fake_mb = self.student(G_sample_mb)
            g_loss = generator_loss(S_fake_mb)

            g_loss.backward()

            G_solver.step()

            # epsilon_hat computation
            curr_list: List = []
            for lidx in range(self.alpha):
                temp_alpha = (alpha[lidx] + np.log(1 / self.delta)) / float(lidx + 1)
                curr_list = curr_list + [temp_alpha]

            epsilon_hat = np.min(curr_list)

        return self

    def sample(self, count: int) -> np.ndarray:
        with torch.no_grad():
            # TODO: fix schema
            Z_mb = self.sample_Z(10 * count, self.z_dim)

            x_hat = self.generator(Z_mb).cpu().numpy()

            return self.encoder.inverse_transform(x_hat)

    # Sample from uniform distribution
    def sample_Z(self, m: int, n: int) -> torch.Tensor:
        return -2 * torch.rand(m, n) + 1

    def _pate_lamda(self, x: np.ndarray, teacher_models: List) -> Tuple[int, int, int]:
        """Returns PATE_lambda(x).

        Args:
          - x: feature vector
          - teacher_models: a list of teacher models

        Returns:
          - n0, n1: the number of label 0 and 1, respectively
          - out: label after adding laplace noise.
        """

        y_hat: List = []

        for teacher in teacher_models:
            temp_y = teacher.predict(np.reshape(x, [1, -1]))
            y_hat = y_hat + [temp_y]

        y_hat_np = np.asarray(y_hat)
        n0 = sum(y_hat_np == 0)
        n1 = sum(y_hat_np == 1)

        lap_noise = np.random.laplace(loc=0.0, scale=self.lamda)

        out = (n1 + lap_noise) / float(n0 + n1)
        out = int(out > 0.5)

        return n0, n1, out


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
        epochs: int = 1,
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
