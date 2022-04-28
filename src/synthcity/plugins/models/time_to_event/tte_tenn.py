"""
TENN: time-to-event prediction using NNs and calibration losses.
"""
# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# synthcity absolute
import synthcity.logger as log
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.models.mlp import MLP
from synthcity.utils.reproducibility import enable_reproducible_results

# synthcity relative
from ._base import TimeToEventPlugin
from .samplers import ImbalancedDatasetSampler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimeEventNN(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_features: int,
        n_layers_hidden: int = 2,
        n_units_hidden: int = 300,
        nonlin: str = "leaky_relu",
        n_iter: int = 2000,
        batch_norm: bool = False,
        dropout: float = 0,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        residual: bool = True,
        opt_betas: tuple = (0.9, 0.999),
        batch_size: int = 250,
        n_iter_print: int = 500,
        seed: int = 0,
        n_iter_min: int = 100,
        clipping_value: int = 0,
        lambda_calibration: float = 1,
        lambda_regression_nc: float = 1,
        lambda_regression_c: float = 1,
    ) -> None:
        super(TimeEventNN, self).__init__()

        self.n_features = n_features
        self.lambda_calibration = lambda_calibration
        self.lambda_regression_nc = lambda_regression_nc
        self.lambda_regression_c = lambda_regression_c

        self.generator = MLP(
            task_type="regression",
            n_units_in=n_features,
            n_units_out=1,  # time to event
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            nonlin=nonlin,
            n_iter=n_iter,
            batch_norm=batch_norm,
            dropout=dropout,
            seed=seed,
            lr=lr,
            residual=residual,
            opt_betas=opt_betas,
        ).to(DEVICE)

        # training
        self.n_iter = n_iter
        self.n_iter_print = n_iter_print
        self.n_iter_min = n_iter_min
        self.batch_size = batch_size
        self.clipping_value = clipping_value

        self.seed = seed
        enable_reproducible_results(seed)

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
    ) -> "TimeEventNN":
        Xt = self._check_tensor(X)
        Tt = self._check_tensor(T)
        Et = self._check_tensor(E)

        self._train(
            Xt,
            Tt,
            Et,
        )

        return self

    def generate(self, X: np.ndarray) -> np.ndarray:
        X = self._check_tensor(X).float()

        return self(X).cpu().numpy()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self.generator.eval()

        with torch.no_grad():
            return self.generator(X).detach().cpu()

    def dataloader(
        self, X: torch.Tensor, T: torch.Tensor, E: torch.Tensor
    ) -> DataLoader:
        dataset = TensorDataset(X, T, E)
        sampler = ImbalancedDatasetSampler(X, T, E)

        return DataLoader(
            dataset, batch_size=self.batch_size, sampler=sampler, pin_memory=False
        )

    def _train_epoch_generator(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        E: torch.Tensor,
    ) -> float:
        # Update the G network
        self.generator.optimizer.zero_grad()

        # Calculate G's loss based on noncensored data
        errG_nc = self._loss_regression_nc(
            X[E == 1], T[E == 1]
        ) + self._loss_calibration(X[E == 1], T[E == 1])

        # Calculate G's loss based on censored data
        errG_c = self._loss_regression_c(X[E == 0], T[E == 0])

        # Calculate total loss
        errG = errG_nc + errG_c

        # Calculate gradients for G
        errG.backward()

        # Update G
        if self.clipping_value > 0:
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), self.clipping_value
            )
        self.generator.optimizer.step()

        # Return loss
        return errG.item()

    def _train_epoch(
        self,
        loader: DataLoader,
    ) -> float:

        G_losses = []

        for i, data in enumerate(loader):
            G_losses.append(
                self._train_epoch_generator(
                    *data,
                )
            )

        return np.mean(G_losses)

    def _train(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        E: torch.Tensor,
    ) -> "TimeEventNN":
        X = self._check_tensor(X).float()
        T = self._check_tensor(T).float()
        E = self._check_tensor(E).long()

        # Load Dataset
        loader = self.dataloader(X, T, E)

        # Train loop
        for i in range(self.n_iter):
            g_loss = self._train_epoch(loader)
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i + 1) % self.n_iter_print == 0:
                log.info(f"[{i}/{self.n_iter}]\tLoss_G: {g_loss}")

        return self

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(DEVICE)
        else:
            return torch.from_numpy(np.asarray(X)).to(DEVICE)

    def _loss_calibration(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        # Evaluate calibration error
        if len(X) == 0:
            return 0

        X = X.to(DEVICE)
        T = T.to(DEVICE)
        pred_T = self.generator(X).squeeze()

        def _inner_dist(arr: torch.Tensor) -> torch.Tensor:
            lhs = arr.view(-1, 1).repeat(1, len(arr))

            return lhs - lhs.T

        inner_T_dist = _inner_dist(T)
        inner_pred_T_dist = _inner_dist(pred_T)

        return self.lambda_calibration * nn.MSELoss()(inner_T_dist, inner_pred_T_dist)

    def _loss_regression_c(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        # Evaluate censored error
        if len(X) == 0:
            return 0

        X = X.to(DEVICE)
        T = T.to(DEVICE)
        fake_T = self.generator(X)

        errG_cen = torch.mean(
            nn.ReLU()(T - fake_T)
        )  # fake_T should be >= T for censored data

        # Calculate G's loss based on this output
        return self.lambda_regression_c * errG_cen

    def _loss_regression_nc(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        # Evaluate noncensored error
        if len(X) == 0:
            return 0

        X = X.to(DEVICE)
        T = T.to(DEVICE)
        fake_T = self.generator(X)

        errG_noncen = nn.MSELoss()(
            fake_T, T
        )  # fake_T should be == T for noncensored data

        return self.lambda_regression_nc * errG_noncen


class TENNTimeToEvent(TimeToEventPlugin):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

        self.kwargs = kwargs

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> "TimeToEventPlugin":
        "Training logic"

        self.model = TimeEventNN(
            n_features=X.shape[1],
            **self.kwargs,
        )

        self.scaler_X = MinMaxScaler()
        enc_X = self.scaler_X.fit_transform(X)

        self.scaler_T = MinMaxScaler()
        enc_T = self.scaler_T.fit_transform(T.values.reshape(-1, 1)).squeeze()

        self.model.fit(enc_X, enc_T, Y)

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(self, X: pd.DataFrame) -> pd.Series:
        "Predict time-to-event"

        self.model.eval()

        enc_X = self.scaler_X.fit_transform(X)

        enc_pred_T = self.model.generate(enc_X)
        nn_time_to_event = self.scaler_T.inverse_transform(enc_pred_T).squeeze()

        return pd.Series(nn_time_to_event, index=X.index)

    @staticmethod
    def name() -> str:
        return "tenn"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="n_layers_hidden", low=1, high=4),
            IntegerDistribution(name="n_units_hidden", low=100, high=300, step=50),
            CategoricalDistribution(name="nonlin", choices=["relu", "leaky_relu"]),
            IntegerDistribution(name="n_iter", low=100, high=3000, step=100),
            FloatDistribution(name="dropout", low=0, high=0.2),
            CategoricalDistribution(name="lr", choices=[1e-2, 1e-3, 1e-4]),
            FloatDistribution(
                name="lambda_calibration",
                low=0,
                high=1,
            ),
            FloatDistribution(
                name="lambda_regression_nc",
                low=0,
                high=1,
            ),
            FloatDistribution(
                name="lambda_regression_c",
                low=0,
                high=1,
            ),
        ]
