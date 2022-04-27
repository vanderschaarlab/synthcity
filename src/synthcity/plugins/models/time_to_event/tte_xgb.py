# stdlib
from typing import Any, List

# third party
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
        **kwargs: Any
    ) -> None:
        super().__init__()
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
            "n_jobs": 4,
        }
        lr_params = {
            "C": 1e-3,
            "max_iter": 10000,
        }

        self.model = XGBSEDebiasedBCE(xgboost_params, lr_params)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> "TimeToEventPlugin":
        "Training logic"

        y = convert_to_structured(T, Y)

        self.model.fit(X, y)
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(self, X: pd.DataFrame) -> pd.Series:
        "Predict time-to-event"

        surv_f = self.model.predict(X)
        return pd.Series(trapz(surv_f.values, surv_f.T.index), index=surv_f.index)

    @staticmethod
    def name() -> str:
        return "survival_xgboost"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="max_depth", low=2, high=6),
            IntegerDistribution(name="min_child_weight", low=0, high=50),
            CategoricalDistribution(name="objective", choices=["aft", "cox"]),
            CategoricalDistribution(
                name="strategy", choices=["weibull", "debiased_bce"]
            ),
        ]
