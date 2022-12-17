# stdlib
import platform
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Optional

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments

# synthcity absolute
from synthcity.metrics.representations.OneClass import OneClassLayer
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.utils.constants import DEVICE
from synthcity.utils.serialization import dataframe_hash, load_from_file, save_to_file


class MetricEvaluator(metaclass=ABCMeta):
    """Base class for all metrics.

    Each derived class must implement the following methods:
        evaluate() - compare two datasets and return a dictionary of metrics.
        direction() - direction of metric (bigger better or smaller better).
        type() - type of the metric.
        name() - name of the metric.

    If any method implementation is missing, the class constructor will fail.

    Constructor Args:
        reduction: str
            The way to aggregate metrics across folds. Default: 'mean'.
        n_histogram_bins: int
            The number of bins used in histogram calculation. Default: 10.
        n_folds: int
            The number of folds in cross validation. Default: 3.
        task_type: str
            The type of downstream task. Default: 'classification'.
        workspace: Path
            The directory to save intermediate models or results. Default: Path("workspace").
        use_cache: bool
            Whether to use cache. If True, it will try to load saved results in workspace directory where possible.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        reduction: str = "mean",
        n_histogram_bins: int = 10,
        n_folds: int = 3,
        task_type: str = "classification",
        random_state: int = 0,
        workspace: Path = Path("workspace"),
        use_cache: bool = True,
        default_metric: Optional[str] = None,
    ) -> None:
        self._reduction = reduction
        self._n_histogram_bins = n_histogram_bins
        self._n_folds = n_folds

        self._task_type = task_type
        self._random_state = random_state
        self._workspace = workspace
        self._use_cache = use_cache
        if default_metric is None:
            default_metric = reduction
        self._default_metric = default_metric

        workspace.mkdir(parents=True, exist_ok=True)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    @abstractmethod
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        ...

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    @abstractmethod
    def evaluate_default(self, X_gt: DataLoader, X_syn: DataLoader) -> float:
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

    def _get_oneclass_model(self, X_gt: np.ndarray) -> OneClassLayer:
        X_hash = dataframe_hash(pd.DataFrame(X_gt))

        cache_file = (
            self._workspace
            / f"sc_metric_cache_model_oneclass_{X_hash}_{platform.python_version()}.bkp"
        )
        if cache_file.exists() and self._use_cache:
            return load_from_file(cache_file)

        model = OneClassLayer(
            input_dim=X_gt.shape[1],
            rep_dim=X_gt.shape[1],
            center=torch.ones(X_gt.shape[1]) * 10,
        )
        model.fit(X_gt)

        save_to_file(cache_file, model)

        return model.to(DEVICE)

    def _oneclass_predict(self, model: OneClassLayer, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(X).float().to(DEVICE)).cpu().detach().numpy()

    def use_cache(self, path: Path) -> bool:
        return path.exists() and self._use_cache
