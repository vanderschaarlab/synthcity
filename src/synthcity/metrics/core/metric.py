# stdlib
from abc import ABCMeta, abstractmethod
from typing import Callable, Dict

# third party
import numpy as np
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader


class MetricEvaluator(metaclass=ABCMeta):
    """Metric interface"""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        reduction: str = "mean",
        n_histogram_bins: int = 10,
        n_folds: int = 3,
        task_type: str = "classification",
        random_state: int = 0,
    ) -> None:
        self._reduction = reduction
        self._n_histogram_bins = n_histogram_bins
        self._n_folds = n_folds

        self._task_type = task_type
        self._random_state = random_state

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    @abstractmethod
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        ...

    @staticmethod
    @abstractmethod
    def direction() -> str:
        ...

    @staticmethod
    @abstractmethod
    def type() -> str:
        ...

    @staticmethod
    @abstractmethod
    def name() -> str:
        ...

    @classmethod
    def fqdn(cls) -> str:
        return f"{cls.type()}.{cls.name()}"

    def reduction(self) -> Callable:
        if self._reduction == "mean":
            return np.mean
        elif self._reduction == "max":
            return np.max
        elif self._reduction == "min":
            return np.min
        else:
            raise ValueError(f"Unknown reduction {self._reduction}")
