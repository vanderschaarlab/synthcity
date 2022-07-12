# stdlib
from typing import Any, List, Optional, Union

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.distribution import CategoricalDistribution, Distribution
from synthcity.plugins.core.models.time_series_survival import get_model_template
from synthcity.plugins.core.models.ts_model import TimeSeriesModel
from synthcity.utils.constants import DEVICE

# synthcity relative
from ._base import TimeToEventPlugin


class TSSurvivalFunctionTimeToEvent(TimeToEventPlugin):
    def __init__(
        self,
        time_points: int = 100,
        base_learner: str = "Transformer",
        device: Any = DEVICE,
        **kwargs: Any
    ) -> None:
        super().__init__()

        survival_model = "dynamic_deephit"
        self.model = get_model_template(survival_model)(
            device=device, rnn_type=base_learner, split=time_points
        )
        self.time_points = 5  # time_points

        self.base_learner = base_learner

        self.target_column = "target_column"
        self.tte_column = "tte_column"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        static: Optional[np.ndarray],
        temporal: np.ndarray,
        temporal_horizons: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
    ) -> "TimeToEventPlugin":
        "Training logic"
        self.model.fit(static, temporal, temporal_horizons, T, E)

        self.time_horizons = np.linspace(
            T.min(), T.max(), self.time_points, dtype=int
        ).tolist()

        surv_fn = self.model.predict(
            static, temporal, temporal_horizons, time_horizons=self.time_horizons
        )
        surv_fn = surv_fn.loc[:, ~surv_fn.columns.duplicated()]

        data = np.concatenate(
            [static, surv_fn.values, np.expand_dims(E, axis=1)], axis=1
        )

        Tlog = np.log(T + 1e-8)

        self.tte_regressor = TimeSeriesModel(
            task_type="regression",
            n_static_units_in=static.shape[-1] if static is not None else 0,
            n_temporal_units_in=temporal[0].shape[-1],
            n_temporal_window=temporal[0].shape[0],
            output_shape=[1],
        ).fit(data, temporal, temporal_horizons, Tlog)

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(
        self,
        static: Optional[np.ndarray],
        temporal: Union[List, np.ndarray],
        temporal_horizons: Union[List, np.ndarray],
    ) -> pd.Series:
        "Predict time-to-event"
        return self.predict_any(
            static, temporal, temporal_horizons, np.ones(len(temporal))
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict_any(
        self,
        static: Optional[np.ndarray],
        temporal: Union[List, np.ndarray],
        temporal_horizons: Union[List, np.ndarray],
        E: np.ndarray,
    ) -> pd.Series:
        "Predict time-to-event or censoring"

        surv_fn = self.model.predict(
            static, temporal, temporal_horizons, time_horizons=self.time_horizons
        )
        surv_fn = surv_fn.loc[:, ~surv_fn.columns.duplicated()]

        data = np.concatenate([static, surv_fn.values, E], axis=1)

        return np.exp(self.tte_regressor.predict(data, temporal, temporal_horizons))

    @staticmethod
    def name() -> str:
        return "ts_survival_function_regression"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            CategoricalDistribution(
                name="survival_model",
                choices=["cox_ph", "dynamic_deephit", "ts_survival_xgboost"],
            ),
        ]
