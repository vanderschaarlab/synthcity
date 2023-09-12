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
        survival_base_learner: str = "Transformer",
        regression_base_learner: str = "Transformer",
        device: Any = DEVICE,
        random_state: int = 0,
        n_layers_hidden: int = 1,
        n_units_hidden: int = 250,
        nonlin: str = "leaky_relu",
        n_iter: int = 500,
        dropout: float = 0,
        lr: float = 1e-3,
        patience: int = 20,
        **kwargs: Any
    ) -> None:
        super().__init__()

        survival_model = "dynamic_deephit"
        self.model = get_model_template(survival_model)(
            device=device,
            rnn_type=survival_base_learner,
            split=time_points,
            n_iter=n_iter,
            lr=lr,
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            dropout=dropout,
            patience=patience,
        )
        self.time_points = 10  # time_points

        self.target_column = "target_column"
        self.tte_column = "tte_column"

        self.rnn_generator_extra_args = {
            "n_static_layers_hidden": n_layers_hidden,
            "n_static_units_hidden": n_units_hidden,
            "n_temporal_layers_hidden": n_layers_hidden,
            "n_temporal_units_hidden": n_units_hidden,
            "mode": regression_base_learner,
            "nonlin": nonlin,
            "n_iter": n_iter,
            "dropout": dropout,
            "random_state": random_state,
            "lr": lr,
            "device": device,
        }

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        static: Optional[np.ndarray],
        temporal: np.ndarray,
        observation_times: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
    ) -> "TimeToEventPlugin":
        "Training logic"
        self.model.fit(static, temporal, observation_times, T, E)

        self.time_horizons = np.linspace(
            T.min(), T.max(), self.time_points, dtype=int
        ).tolist()

        surv_fn = self.model.predict(
            static, temporal, observation_times, time_horizons=self.time_horizons
        )
        surv_fn = surv_fn.loc[:, ~surv_fn.columns.duplicated()]

        data = np.concatenate(
            [static, surv_fn.values, np.expand_dims(E, axis=1)], axis=1
        )

        Tlog = np.log(T + 1e-8)

        self.tte_regressor = TimeSeriesModel(
            task_type="regression",
            n_static_units_in=data.shape[-1],
            n_temporal_units_in=temporal[0].shape[-1],
            n_temporal_window=temporal[0].shape[0],
            output_shape=[1],
            **self.rnn_generator_extra_args,
        ).fit(data, temporal, observation_times, Tlog)

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(
        self,
        static: Optional[np.ndarray],
        temporal: Union[List, np.ndarray],
        observation_times: Union[List, np.ndarray],
    ) -> pd.Series:
        "Predict time-to-event"
        return self.predict_any(
            static, temporal, observation_times, np.ones(len(temporal))
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict_any(
        self,
        static: Optional[np.ndarray],
        temporal: Union[List, np.ndarray],
        observation_times: Union[List, np.ndarray],
        E: np.ndarray,
    ) -> pd.Series:
        "Predict time-to-event or censoring"

        surv_fn = self.model.predict(
            static, temporal, observation_times, time_horizons=self.time_horizons
        )
        surv_fn = surv_fn.loc[:, ~surv_fn.columns.duplicated()]

        data = np.concatenate(
            [static, surv_fn.values, np.expand_dims(E, axis=1)], axis=1
        )

        return np.exp(
            self.tte_regressor.predict(data, temporal, observation_times)
        ).squeeze()

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
