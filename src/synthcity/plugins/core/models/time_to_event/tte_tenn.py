"""
TENN: time-to-event prediction using NNs and calibration losses.
"""
# stdlib
from typing import Any, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments
from sklearn.model_selection import train_test_split
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
from synthcity.plugins.core.models.mlp import MLP
from synthcity.plugins.core.models.time_to_event.metrics import (
    c_index,
    expected_time_error,
)
from synthcity.utils.constants import DEVICE
from synthcity.utils.reproducibility import enable_reproducible_results
from synthcity.utils.samplers import ImbalancedDatasetSampler

# synthcity relative
from ._base import TimeToEventPlugin


class TimeEventNN(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_features: int,
        n_layers_hidden: int = 1,
        n_units_hidden: int = 350,
        nonlin: str = "leaky_relu",
        n_iter: int = 1000,
        batch_norm: bool = False,
        dropout: float = 0.02,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        residual: bool = True,
        opt_betas: tuple = (0.9, 0.999),
        batch_size: int = 500,
        n_iter_print: int = 50,
        random_state: int = 0,
        n_iter_min: int = 100,
        clipping_value: int = 0,
        patience: int = 5,
        lambda_calibration: float = 1,
        lambda_ordering: float = 1,
        lambda_regression_nc: float = 1,
        lambda_regression_c: float = 1,
        device: Any = DEVICE,
    ) -> None:
        super(TimeEventNN, self).__init__()

        self.device = device
        self.n_features = n_features
        self.lambda_calibration = lambda_calibration
        self.lambda_ordering = lambda_ordering
        self.lambda_regression_nc = lambda_regression_nc
        self.lambda_regression_c = lambda_regression_c
        self.patience = patience

        self.generator = MLP(
            task_type="regression",
            n_units_in=n_features,
            n_units_out=1,  # time to event
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            nonlin=nonlin,
            nonlin_out=[("relu", 1)],
            n_iter=n_iter,
            batch_norm=batch_norm,
            dropout=dropout,
            random_state=random_state,
            lr=lr,
            residual=residual,
            opt_betas=opt_betas,
            device=device,
        ).to(self.device)

        # training
        self.n_iter = n_iter
        self.n_iter_print = n_iter_print
        self.n_iter_min = n_iter_min
        self.batch_size = batch_size
        self.clipping_value = clipping_value

        self.random_state = random_state
        enable_reproducible_results(random_state)

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
        self.generator.eval()
        X = self._check_tensor(X).float()

        with torch.no_grad():
            return self(X).cpu().numpy()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.generator(X)

    def dataloader(
        self, X: torch.Tensor, T: torch.Tensor, E: torch.Tensor
    ) -> Tuple[DataLoader, TensorDataset]:
        X_train, X_val, T_train, T_val, E_train, E_val = train_test_split(
            X.cpu(),
            T.cpu(),
            E.cpu(),
            stratify=E.cpu(),
            random_state=self.random_state,
        )

        train_dataset = TensorDataset(
            self._check_tensor(X_train),
            self._check_tensor(T_train),
            self._check_tensor(E_train),
        )
        val_dataset = TensorDataset(
            self._check_tensor(X_val),
            self._check_tensor(T_val),
            self._check_tensor(E_val),
        )

        sampler = ImbalancedDatasetSampler(E_train.cpu().numpy().tolist())

        return DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=sampler, pin_memory=False
        ), DataLoader(val_dataset, batch_size=len(val_dataset), pin_memory=False)

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

        # Calculate ordering loss
        errG_order = self._loss_ordering(X, T, E)

        # Calculate total loss
        errG = errG_nc + errG_c + errG_order

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
            Xmb, Tmb, Emb = data
            Tmb, idxs = torch.sort(Tmb)
            Xmb = Xmb[idxs]
            Emb = Emb[idxs]

            G_losses.append(
                self._train_epoch_generator(
                    Xmb,
                    Tmb,
                    Emb,
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
        loader, val_loader = self.dataloader(X, T, E)

        best_exp_tte_err = 9999
        patience = 0

        # Train loop
        for i in range(self.n_iter):
            g_loss = self._train_epoch(loader)
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i + 1) % self.n_iter_print == 0:
                with torch.no_grad():
                    Xval, Tval, Eval = next(iter(val_loader))
                    Tdf = pd.Series(Tval.cpu().numpy())
                    Edf = pd.Series(Eval.cpu().numpy(), index=Tdf.index)

                    Tpred = pd.Series(
                        self.generator(Xval).detach().cpu().numpy().squeeze(),
                        index=Tdf.index,
                    )

                    c_index_val = c_index(Tdf, Edf, Tpred)
                    exp_err_val = expected_time_error(Tdf, Edf, Tpred)

                    if best_exp_tte_err > exp_err_val:
                        patience = 0
                        best_exp_tte_err = exp_err_val
                    else:
                        patience += 1

                    if patience > self.patience:
                        log.debug(
                            f"No improvement for {patience} iterations. stopping..."
                        )
                        break

                log.debug(
                    f"[{i}/{self.n_iter}]\tTrain loss: {g_loss} C-Index val: {c_index_val} Expected time err: {exp_err_val}"
                )

        return self

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)

    def _loss_calibration(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        # Evaluate calibration error
        if len(X) <= 1:
            return 0

        pred_T = self.generator(X).squeeze()

        def _inner_dist(arr: torch.Tensor) -> torch.Tensor:
            lhs = arr.view(-1, 1).repeat(1, len(arr))

            return lhs - lhs.T

        inner_T_dist = _inner_dist(T)
        inner_pred_T_dist = _inner_dist(pred_T)

        err = self.lambda_calibration * nn.MSELoss()(inner_T_dist, inner_pred_T_dist)

        if torch.isnan(err):
            raise RuntimeError("Calibration loss contains NaNs")

        return err

    def _loss_ordering(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        E: torch.Tensor,
    ) -> torch.Tensor:
        # Evaluate calibration error
        if len(X) <= 1:
            return 0

        pred_T = self.generator(X).squeeze()

        def _inner_dist(arr: torch.Tensor) -> torch.Tensor:
            arr_event = arr[E == 1]
            lhs = arr.view(-1, 1).repeat(1, len(arr_event))

            return nn.ReLU()(
                lhs - arr_event
            )  # we only want the points after each event, not before

        inner_T_dist = _inner_dist(T)
        inner_pred_T_dist = _inner_dist(pred_T)

        return self.lambda_ordering * nn.MSELoss()(inner_pred_T_dist, inner_T_dist)

    def _loss_ordering_v2(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        E: torch.Tensor,
    ) -> torch.Tensor:
        # Evaluate ranking error.
        # T is expected to be ordered ascending.
        if len(X) <= 1:
            return torch.tensor(0).to(self.device)
        pred_T = self.generator(X).squeeze()
        err = torch.tensor(0).float().to(self.device)

        for idx in range(1, len(pred_T)):
            prev_T = pred_T[E == 1][:idx]
            fails = prev_T[prev_T[:idx] > pred_T[idx]]

            if len(fails) == 0:
                continue

            err += nn.MSELoss()(fails, pred_T[idx])

        if torch.isnan(err):
            raise RuntimeError("Ranking loss contains NaNs")

        return self.lambda_ordering * err

    def _loss_regression_c(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        # Evaluate censored error
        if len(X) == 0:
            return 0

        fake_T = self.generator(X)

        errG_cen = torch.mean(
            nn.ReLU()(T - fake_T)
        )  # fake_T should be >= T for censored data

        if torch.isnan(errG_cen):
            raise RuntimeError("Censored regression loss contains NaNs")
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

        fake_T = self.generator(X).squeeze()

        errG_noncen = nn.MSELoss()(
            fake_T, T
        )  # fake_T should be == T for noncensored data

        if torch.isnan(errG_noncen):
            raise RuntimeError("Observed regression loss contains NaNs")
        return self.lambda_regression_nc * errG_noncen


class TENNTimeToEvent(TimeToEventPlugin):
    def __init__(
        self, model_search_n_iter: Optional[int] = None, **kwargs: Any
    ) -> None:
        super().__init__()

        if model_search_n_iter is not None:
            kwargs["n_iter"] = model_search_n_iter

        self.kwargs = kwargs

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> "TimeToEventPlugin":
        "Training logic"
        self._fit_censoring_model(X, T, Y)

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

        enc_pred_T = self.model.generate(enc_X).reshape(-1, 1)
        nn_time_to_event = self.scaler_T.inverse_transform(enc_pred_T).squeeze()

        return pd.Series(nn_time_to_event, index=X.index)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict_any(self, X: pd.DataFrame, E: pd.Series) -> pd.Series:
        "Predict time-to-event"

        result = pd.Series([0] * len(X), index=E.index)

        if (E == 1).sum() > 0:
            result[E == 1] = self.predict(X[E == 1])
        if (E == 0).sum() > 0:
            result[E == 0] = self._predict_censoring(X[E == 0])

        return result

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
            CategoricalDistribution(
                name="lambda_calibration",
                choices=[
                    0,
                    1,
                    10,
                    50,
                    100,
                ],
            ),
            CategoricalDistribution(
                name="lambda_regression_nc",
                choices=[
                    0,
                    1,
                    10,
                    50,
                    100,
                ],
            ),
            CategoricalDistribution(
                name="lambda_regression_c",
                choices=[
                    0,
                    1,
                    10,
                    50,
                    100,
                ],
            ),
            CategoricalDistribution(
                name="lambda_ordering",
                choices=[
                    0,
                    1,
                    10,
                    50,
                    100,
                ],
            ),
        ]
