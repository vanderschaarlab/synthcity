# stdlib
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from scipy.integrate import trapz
from sksurv.ensemble import RandomSurvivalForest

# synthcity absolute
from synthcity.plugins.core.distribution import Distribution, IntegerDistribution

# synthcity relative
from ._base import TimeToEventPlugin


class RandomSurvivalForestTimeToEvent(TimeToEventPlugin):
    def __init__(
        self, model_search_n_iter: Optional[int] = None, **kwargs: Any
    ) -> None:
        super().__init__()

        if model_search_n_iter is not None:
            kwargs["n_estimators"] = 10 * model_search_n_iter

        self.model = RandomSurvivalForest(max_depth=3, **kwargs)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> "TimeToEventPlugin":
        "Training logic"

        y = [(Y.iloc[i], T.iloc[i]) for i in range(len(X))]
        y = np.array(y, dtype=[("status", "bool"), ("time", "<f8")])

        self.model.fit(np.asarray(X), y)

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(self, X: pd.DataFrame) -> pd.Series:
        "Predict time-to-event"

        expectations = []
        for surv_f in self.model.predict_survival_function(X):
            expectations.append(trapz(surv_f.y, surv_f.x))

        return pd.Series(expectations, index=X.index)

    @staticmethod
    def name() -> str:
        return "random_survival_forest"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="n_estimators", low=10, high=100),
        ]
