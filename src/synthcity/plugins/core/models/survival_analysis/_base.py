# stdlib
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List

# third party
import pandas as pd
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.serializable import Serializable


class SurvivalAnalysisPlugin(Serializable, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> Any:
        "Training logic"
        ...

    @abstractmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(self, X: pd.DataFrame, time_horizons: List) -> pd.DataFrame:
        "Predict risk"
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
