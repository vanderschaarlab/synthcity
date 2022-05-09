# stdlib
from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, List, Optional

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
        n_folds: int = 3,
        task_type: str = "classification",
        target_column: Optional[str] = None,
        time_to_event_column: Optional[str] = None,
        time_horizons: Optional[List] = None,
        random_seed: int = 0,
    ) -> None:
        self._sensitive_columns = sensitive_columns
        self._reduction = reduction
        self._n_histogram_bins = n_histogram_bins
        self._n_folds = n_folds

        self._task_type = task_type
        self._target_column = target_column
        self._time_to_event_column = time_to_event_column
        self._time_horizons = time_horizons
        self._random_seed = random_seed

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    @abstractmethod
    def evaluate(
        self, X_gt_train: pd.DataFrame, X_gt_test: pd.DataFrame, X_syn: pd.DataFrame
    ) -> Dict:
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
