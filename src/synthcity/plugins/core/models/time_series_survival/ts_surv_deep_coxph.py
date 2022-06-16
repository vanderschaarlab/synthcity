# stdlib
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd
from auton_survival.models.cph import DeepRecurrentCoxPH
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    IntegerDistribution,
)
from synthcity.utils.constants import DEVICE
from synthcity.utils.reproducibility import enable_reproducible_results

# synthcity relative
from ._base import TimeSeriesSurvivalPlugin


class DeepCoxPHTimeSeriesSurvival(TimeSeriesSurvivalPlugin):
    def __init__(
        self,
        n_iter: int = 1000,
        batch_size: int = 100,
        lr: float = 1e-3,
        n_layers_hidden: int = 2,
        n_units_hidden: int = 300,
        seed: int = 0,
        device: Any = DEVICE,
        **kwargs: Any
    ) -> None:
        super().__init__()
        enable_reproducible_results(seed)

        self.lr = lr
        self.batch_size = batch_size
        self.n_iter = n_iter

        self.model = DeepRecurrentCoxPH(
            layers=n_layers_hidden,
            hidden=n_units_hidden,
        )

    def _merge_data(
        self,
        static: Optional[np.ndarray],
        temporal: np.ndarray,
    ) -> np.ndarray:
        if static is None:
            return temporal

        merged = []
        for idx, item in enumerate(temporal):
            local_static = static[idx].reshape(1, -1)
            local_static = np.repeat(local_static, len(temporal[idx]), axis=0)
            tst = np.concatenate([temporal[idx], local_static], axis=1)
            merged.append(tst)

        return np.array(merged, dtype=object)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        static: Optional[np.ndarray],
        temporal: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
    ) -> TimeSeriesSurvivalPlugin:
        data = self._merge_data(static, temporal)
        self.model.fit(
            data,
            T,
            E,
            batch_size=self.batch_size,
            learning_rate=self.lr,
            iters=self.n_iter,
        )
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(
        self,
        static: Optional[np.ndarray],
        temporal: np.ndarray,
        time_horizons: List,
    ) -> np.ndarray:
        "Predict risk"

        data = self._merge_data(static, temporal)
        raw = self.model.predict_risk(data, time_horizons)
        out = []
        offset = -1
        for item in temporal:
            offset += len(item)
            out.append(raw[offset])
        return pd.DataFrame(out, columns=time_horizons)

    @staticmethod
    def name() -> str:
        return "deep_recurrent_coxph"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="n_units_hidden", low=10, high=100, step=10),
            IntegerDistribution(name="n_layers_hidden", low=1, high=4),
            CategoricalDistribution(name="batch_size", choices=[100, 200, 500]),
            CategoricalDistribution(name="lr", choices=[1e-2, 1e-3, 1e-4]),
        ]
