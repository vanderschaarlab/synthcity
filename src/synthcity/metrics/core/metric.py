# stdlib
from abc import ABCMeta, abstractmethod
from typing import Callable, List

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments


class MetricEvaluator(metaclass=ABCMeta):
    """Metric interface"""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        sensitive_columns: List[str] = [],
        reduction: str = "mean",
        n_histogram_bins: int = 10,
    ) -> None:
        self._sensitive_columns = sensitive_columns
        self._reduction = reduction
        self._n_histogram_bins = n_histogram_bins

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    @abstractmethod
    def evaluate(self, X_gt: pd.DataFrame, X_syn: pd.DataFrame) -> float:
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
        raise NotImplementedError()

    def reduction(self) -> Callable:
        if self._reduction == "mean":
            return np.mean
        elif self._reduction == "max":
            return np.max
        elif self._reduction == "min":
            return np.min
        else:
            raise ValueError(f"Unknown reduction {self._reduction}")
