# stdlib
from typing import Any, List, Tuple

# third party
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import DeepHitSingle
from pydantic import validate_arguments
from sklearn.model_selection import train_test_split

# synthcity absolute
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.utils.constants import DEVICE
from synthcity.utils.reproducibility import enable_reproducible_results

# synthcity relative
from ._base import SurvivalAnalysisPlugin


class DeephitSurvivalAnalysis(SurvivalAnalysisPlugin):
    def __init__(
        self,
        num_durations: int = 500,
        batch_size: int = 100,
        epochs: int = 2000,
        lr: float = 1e-2,
        dim_hidden: int = 300,
        alpha: float = 0.28,
        sigma: float = 0.38,
        dropout: float = 0.2,
        patience: int = 20,
        batch_norm: bool = False,
        random_state: int = 0,
        device: Any = DEVICE,
        **kwargs: Any
    ) -> None:
        super().__init__()
        enable_reproducible_results(random_state)

        self.device = device
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
    def fit(
        self, X: pd.DataFrame, T: pd.Series, E: pd.Series
    ) -> "SurvivalAnalysisPlugin":
        if (E.unique() == [0]).all():
            raise RuntimeError("The input contains only censored data")

        labtrans = DeepHitSingle.label_transform(self.num_durations)

        X = np.asarray(X).astype("float32")

        X_train, X_val, E_train, E_val, T_train, T_val = train_test_split(
            X, E, T, random_state=42
        )

        def get_target(df: Any) -> Tuple:
            return (np.asarray(df[0]), np.asarray(df[1]))

        y_train = labtrans.fit_transform(*get_target((T_train, E_train)))
        y_val = labtrans.transform(*get_target((T_val, E_val)))
        self.duration_index = labtrans.cuts

        in_features = X_train.shape[1]
        out_features = labtrans.out_features

        self.net = torch.nn.Sequential(
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

        training_model = DeepHitSingle(
            self.net,
            tt.optim.Adam,
            alpha=self.alpha,
            sigma=self.sigma,
            duration_index=self.duration_index,
        )
        training_model.optimizer.set_lr(self.lr)

        callbacks = [tt.callbacks.EarlyStopping(patience=self.patience)]
        training_model.fit(
            X_train,
            y_train,
            self.batch_size,
            self.epochs,
            callbacks,
            val_data=(X_val, y_val),
            verbose=False,
        )

        self.net.eval()
        self.model = DeepHitSingle(
            self.net,
            alpha=self.alpha,
            sigma=self.sigma,
            duration_index=self.duration_index,
        )

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(self, X: pd.DataFrame, time_horizons: List) -> pd.DataFrame:
        "Predict risk"

        Xnp = np.asarray(X).astype("float32")
        surv = self.model.predict_surv_df(Xnp).T

        preds_ = np.zeros([np.shape(surv)[0], len(time_horizons)])

        time_bins = surv.columns
        for t, eval_time in enumerate(time_horizons):
            nearest = self._find_nearest(time_bins, eval_time)
            preds_[:, t] = np.asarray(1 - surv[nearest])

        return pd.DataFrame(preds_, columns=time_horizons, index=X.index)

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

    def _find_nearest(self, array: np.ndarray, value: float) -> float:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
