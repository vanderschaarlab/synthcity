# stdlib
from typing import Any, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import DeepHitSingle
from pydantic import validate_arguments
from scipy.integrate import trapz
from sklearn.model_selection import train_test_split

# synthcity absolute
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.utils.constants import DEVICE

# synthcity relative
from ._base import TimeToEventPlugin


class DeephitTimeToEvent(TimeToEventPlugin):
    def __init__(
        self,
        model_search_n_iter: Optional[int] = None,
        num_durations: int = 1000,
        batch_size: int = 100,
        epochs: int = 2000,
        lr: float = 1e-3,
        dim_hidden: int = 300,
        alpha: float = 0.28,
        sigma: float = 0.38,
        dropout: float = 0.02,
        patience: int = 20,
        batch_norm: bool = False,
        device: Any = DEVICE,
        **kwargs: Any
    ) -> None:
        super().__init__()

        self.device = device
        if model_search_n_iter is not None:
            epochs = model_search_n_iter

        self.num_durations = num_durations
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.dim_hidden = dim_hidden
        self.alpha = alpha
        self.sigma = sigma
        self.patience = patience
        self.dropout = dropout
        self.batch_norm = batch_norm

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: pd.DataFrame, T: pd.Series, E: pd.Series) -> "TimeToEventPlugin":
        self._fit_censoring_model(X, T, E)

        labtrans = DeepHitSingle.label_transform(self.num_durations)
        X = np.asarray(X).astype("float32")
        T = np.asarray(T).astype(int)

        X_train, X_val, E_train, E_val, T_train, T_val = train_test_split(
            X, E, T, random_state=42
        )

        def get_target(df: Any) -> Tuple:
            return (np.asarray(df[0]), np.asarray(df[1]))

        y_train = labtrans.fit_transform(*get_target((T_train, E_train)))
        y_val = labtrans.transform(*get_target((T_val, E_val)))

        in_features = X_train.shape[1]
        out_features = labtrans.out_features

        net = torch.nn.Sequential(
            torch.nn.Linear(in_features, self.dim_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.dim_hidden, self.dim_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.dim_hidden, self.dim_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.dim_hidden, out_features),
        ).to(self.device)

        self.model = DeepHitSingle(
            net,
            tt.optim.Adam,
            alpha=self.alpha,
            sigma=self.sigma,
            duration_index=labtrans.cuts,
        )

        self.model.optimizer.set_lr(self.lr)

        callbacks = [tt.callbacks.EarlyStopping(patience=self.patience)]
        self.model.fit(
            X_train,
            y_train,
            self.batch_size,
            self.epochs,
            callbacks,
            val_data=(X_val, y_val),
            verbose=False,
        )

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(self, X: pd.DataFrame) -> pd.Series:
        "Predict time-to-event"
        self.model.net.eval()

        X_np = np.asarray(X).astype("float32")

        surv_f = self.model.predict_surv_df(X_np)

        return pd.Series(trapz(surv_f.T.values, surv_f.index.values), index=X.index)

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
        return "deephit"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        return [
            CategoricalDistribution(name="batch_size", choices=[100, 200, 500]),
            CategoricalDistribution(name="lr", choices=[1e-2, 1e-3, 1e-4]),
            IntegerDistribution(name="dim_hidden", low=10, high=100, step=10),
            FloatDistribution(name="alpha", low=0.0, high=0.5),
            FloatDistribution(name="sigma", low=0.0, high=0.5),
            FloatDistribution(name="dropout", low=0.0, high=0.2),
            IntegerDistribution(name="patience", low=10, high=50),
        ]
