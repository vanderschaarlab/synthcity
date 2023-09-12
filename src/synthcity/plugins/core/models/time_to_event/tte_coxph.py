# stdlib
from typing import Any, List, Optional

# third party
import pandas as pd
from lifelines import CoxPHFitter
from pydantic import validate_arguments
from scipy.integrate import trapz

# synthcity absolute
from synthcity.plugins.core.distribution import Distribution, FloatDistribution
from synthcity.utils.constants import DEVICE

# synthcity relative
from ._base import TimeToEventPlugin


class CoxPHTimeToEvent(TimeToEventPlugin):
    def __init__(
        self,
        model_search_n_iter: Optional[int] = None,
        device: Any = DEVICE,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.model = CoxPHFitter(**kwargs)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> "TimeToEventPlugin":
        "Training logic"
        self._fit_censoring_model(X, T, Y)

        df = X.copy()
        df["event"] = Y
        df["time"] = T

        self.model.fit(df, "time", "event")

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(self, X: pd.DataFrame) -> pd.Series:
        "Predict time-to-event"

        surv_f = self.model.predict_survival_function(X)

        return pd.Series(trapz(surv_f.values.T, surv_f.index), index=surv_f.T.index)

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
        return "cox_ph"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            FloatDistribution(name="alpha", low=0.0, high=0.1),
            FloatDistribution(name="penalizer", low=0, high=0.2),
        ]
