# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.distribution import Distribution, FloatDistribution
from synthcity.utils.constants import DEVICE

# synthcity relative
from ._base import SurvivalAnalysisPlugin


class WeibullAFTSurvivalAnalysis(SurvivalAnalysisPlugin):
    def __init__(self, device: Any = DEVICE, **kwargs: Any) -> None:
        super().__init__()
        self.model = WeibullAFTFitter(**kwargs)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self, X: pd.DataFrame, T: pd.Series, Y: pd.Series
    ) -> "SurvivalAnalysisPlugin":
        "Training logic"

        df = X.copy()
        df["event"] = Y
        df["time"] = T

        self.model.fit(df, "time", "event")

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(self, X: pd.DataFrame, time_horizons: List) -> pd.DataFrame:
        "Predict time-to-event"
        chunks = int(len(X) / 1024) + 1

        preds_ = []
        for chunk in np.array_split(X, chunks):
            local_preds_ = np.zeros([len(chunk), len(time_horizons)])
            surv = self.model.predict_survival_function(chunk)
            surv_times = np.asarray(surv.index).astype(int)
            surv = np.asarray(surv.T)

            for t, eval_time in enumerate(time_horizons):
                tmp_time = np.where(eval_time <= surv_times)[0]
                if len(tmp_time) == 0:
                    local_preds_[:, t] = 1.0 - surv[:, 0]
                else:
                    local_preds_[:, t] = 1.0 - surv[:, tmp_time[0]]

            preds_.append(local_preds_)

        return pd.DataFrame(
            np.concatenate(preds_, axis=0), columns=time_horizons, index=X.index
        )

    @staticmethod
    def name() -> str:
        return "weibull_aft"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            FloatDistribution(name="alpha", low=0.0, high=0.1),
            FloatDistribution(name="l1_ratio", low=0, high=0.2),
        ]
