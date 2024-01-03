# stdlib
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from xgboost import XGBRegressor

# synthcity absolute
from synthcity.plugins.core.distribution import CategoricalDistribution, Distribution
from synthcity.plugins.core.models.survival_analysis import get_model_template
from synthcity.utils.constants import DEVICE

# synthcity relative
from ._base import TimeToEventPlugin


class SurvivalFunctionTimeToEvent(TimeToEventPlugin):
    def __init__(
        self,
        time_points: int = 100,
        model_search_n_iter: Optional[int] = None,
        device: Any = DEVICE,
        **kwargs: Any
    ) -> None:
        super().__init__()

        survival_model = "deephit"
        self.model = get_model_template(survival_model)(device=device)
        self.time_points = time_points
        self.target_column = "target_column"
        self.tte_column = "tte_column"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> "TimeToEventPlugin":
        "Training logic"
        self.model.fit(X, T, Y)

        self.time_horizons = np.linspace(
            T.min(), T.max(), self.time_points, dtype=int
        ).tolist()

        surv_fn = self.model.predict(X, time_horizons=self.time_horizons)
        surv_fn = surv_fn.loc[:, ~surv_fn.columns.duplicated()]

        data = X.copy()
        data[surv_fn.columns] = surv_fn
        data[self.target_column] = Y

        Tlog = np.log(T + 1e-8)

        xgb_params = {
            "n_jobs": 2,
            "n_estimators": 100,
            "verbosity": 0,
            "depth": 3,
            "random_state": 0,
        }

        self.tte_regressor = XGBRegressor(**xgb_params).fit(data, Tlog)

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

        data = X.copy()
        data[surv_fn.columns] = surv_fn
        data[self.target_column] = E

        return np.exp(self.tte_regressor.predict(data))

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
