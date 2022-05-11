# stdlib
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from xgboost import XGBRegressor

# synthcity absolute
from synthcity.plugins.core.distribution import CategoricalDistribution, Distribution
from synthcity.plugins.models.survival_analysis import get_model_template

# synthcity relative
from ._base import TimeToEventPlugin


class SurvivalFunctionTimeToEvent(TimeToEventPlugin):
    def __init__(
        self,
        survival_model: str = "survival_xgboost",
        time_points: int = 10,
        model_search_n_iter: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()

        self.model = get_model_template(survival_model)(**kwargs)
        self.time_points = time_points
        self.target_column = "target_column"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> "TimeToEventPlugin":
        "Training logic"

        self.model.fit(X, T, Y)
        self.time_horizons = np.linspace(
            T.min(), T.max(), self.time_points, dtype=int
        ).tolist()

        surv_fn = self.model.predict(X, time_horizons=self.time_horizons)
        surv_fn = surv_fn.loc[:, ~surv_fn.columns.duplicated()]

        surv_fn[self.target_column] = Y

        xgb_params = {
            "n_jobs": 2,
            "verbosity": 0,
            "depth": 5,
            "random_state": 0,
        }

        self.tte_regressor = XGBRegressor(**xgb_params).fit(surv_fn, T)

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(self, X: pd.DataFrame) -> pd.Series:
        "Predict time-to-event"
        return self.predict_any(X, pd.Series([1] * len(X), index=X.index))

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict_any(self, X: pd.DataFrame, E: pd.Series) -> pd.Series:
        "Predict time-to-event or censoring"

        surv_fn = self.model.predict(X, time_horizons=self.time_horizons)
        surv_fn = surv_fn.loc[:, ~surv_fn.columns.duplicated()]

        surv_fn[self.target_column] = E

        return self.tte_regressor.predict(surv_fn)

    @staticmethod
    def name() -> str:
        return "survival_function_regression"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            CategoricalDistribution(
                name="survival_model",
                choices=["weibull_aft", "cox_ph", "deephit", "survival_xgboost"],
            ),
        ]
