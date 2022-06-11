# stdlib
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List

# third party
import pandas as pd
from pydantic import validate_arguments
from xgboost import XGBRegressor

# synthcity absolute
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.serializable import Serializable


class TimeToEventPlugin(Serializable, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> Any:
        "Training logic"
        ...

    @abstractmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(self, X: pd.DataFrame) -> pd.Series:
        "Predict time-to-event"
        ...

    @abstractmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict_any(self, X: pd.DataFrame, E: pd.Series) -> pd.Series:
        "Predict time-to-event or censoring"
        ...

    @staticmethod
    @abstractmethod
    def name() -> str:
        """The name of the plugin."""
        ...

    @staticmethod
    @abstractmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        """Returns the hyperparameter space for the derived plugin."""
        ...

    @classmethod
    def sample_hyperparameters(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Sample value from the hyperparameter space for the current plugin."""
        param_space = cls.hyperparameter_space(*args, **kwargs)

        results = {}

        for hp in param_space:
            results[hp.name] = hp.sample()[0]

        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _fit_censoring_model(self, X: pd.DataFrame, T: pd.Series, E: pd.Series) -> Any:
        xgb_params = {
            "n_jobs": 2,
            "verbosity": 0,
            "depth": 5,
            "random_state": 0,
        }

        self.tte_regressor = XGBRegressor(**xgb_params).fit(X[E == 0], T[E == 0])

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _predict_censoring(self, X: pd.DataFrame) -> Any:
        return self.tte_regressor.predict(X)
