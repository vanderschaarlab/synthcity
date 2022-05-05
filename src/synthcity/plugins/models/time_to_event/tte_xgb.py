# stdlib
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from scipy.integrate import trapz
from xgbse import XGBSEDebiasedBCE
from xgbse.converters import convert_to_structured

# synthcity absolute
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    IntegerDistribution,
)

# synthcity relative
from ._base import TimeToEventPlugin


class XGBTimeToEvent(TimeToEventPlugin):
    booster = ["gbtree", "gblinear", "dart"]

    def __init__(
        self,
        model_search_n_iter: Optional[int] = None,
        n_estimators: int = 100,
        colsample_bynode: float = 0.5,
        max_depth: int = 8,
        subsample: float = 0.5,
        learning_rate: float = 5e-2,
        min_child_weight: int = 50,
        tree_method: str = "hist",
        booster: int = 2,
        random_state: int = 0,
        objective: str = "aft",  # "aft", "cox"
        strategy: str = "weibull",  # "weibull", "debiased_bce"
        time_points: int = 10,
        **kwargs: Any
    ) -> None:
        super().__init__()

        if model_search_n_iter is not None:
            n_estimators = 10 * model_search_n_iter

        surv_params = {}
        if objective == "aft":
            surv_params = {
                "objective": "survival:aft",
                "eval_metric": "aft-nloglik",
                "aft_loss_distribution": "normal",
                "aft_loss_distribution_scale": 1.0,
            }
        else:
            surv_params = {
                "objective": "survival:cox",
                "eval_metric": "cox-nloglik",
            }
        xgboost_params = {
            # survival
            **surv_params,
            **kwargs,
            # basic xgboost
            "n_estimators": n_estimators,
            "colsample_bynode": colsample_bynode,
            "max_depth": max_depth,
            "subsample": subsample,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "verbosity": 0,
            "tree_method": tree_method,
            "booster": XGBTimeToEvent.booster[booster],
            "random_state": random_state,
            "n_jobs": 2,
        }
        lr_params = {
            "C": 1e-3,
            "max_iter": 10000,
        }

        self.model = XGBSEDebiasedBCE(xgboost_params, lr_params)
        self.time_points = time_points

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> "TimeToEventPlugin":
        "Training logic"

        y = convert_to_structured(T, Y)

        censored_times = T[Y == 0]
        obs_times = T[Y == 1]

        lower_bound = max(censored_times.min(), obs_times.min()) + 1
        if pd.isna(lower_bound):
            lower_bound = T.min()
        upper_bound = min(censored_times.max(), obs_times.max()) - 1
        if pd.isna(upper_bound):
            upper_bound = T.max()

        time_bins = np.linspace(lower_bound, upper_bound, self.time_points, dtype=int)

        self.model.fit(X, y, time_bins=time_bins)

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(self, X: pd.DataFrame) -> pd.Series:
        "Predict time-to-event"

        surv_f = self.model.predict(X)
        return pd.Series(trapz(surv_f.values, surv_f.T.index), index=X.index)

    @staticmethod
    def name() -> str:
        return "survival_xgboost"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="max_depth", low=2, high=4),
            IntegerDistribution(name="n_estimators", low=5, high=100),
            IntegerDistribution(name="min_child_weight", low=0, high=50),
            CategoricalDistribution(name="objective", choices=["aft", "cox"]),
            CategoricalDistribution(
                name="strategy", choices=["weibull", "debiased_bce"]
            ),
        ]
