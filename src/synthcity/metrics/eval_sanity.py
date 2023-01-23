# stdlib
from typing import Any, Dict

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.neighbors import NearestNeighbors

# synthcity absolute
from synthcity.metrics.core import MetricEvaluator
from synthcity.plugins.core.dataloader import DataLoader


class BasicMetricEvaluator(MetricEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_sanity.BasicMetricEvaluator
        :parts: 1
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def _helper_nearest_neighbor(X_gt: DataLoader, X_syn: DataLoader) -> np.ndarray:
        try:
            estimator = NearestNeighbors(n_neighbors=5).fit(X_syn.numpy())
            dist, _ = estimator.kneighbors(X_gt.numpy(), 1, return_distance=True)
            return dist.squeeze()
        except BaseException:
            return np.asarray([999])

    @staticmethod
    def type() -> str:
        return "sanity"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_default(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> float:
        return self.evaluate(X_gt, X_syn)[self._default_metric]


class DataMismatchScore(BasicMetricEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_sanity.DataMismatchScore
        :parts: 1

    Basic sanity score. Compares the data types between the column of the ground truth and the synthetic data.

    Score:
        0: no datatype mismatch.
        1: complete data type mismatch between the datasets.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)

    @staticmethod
    def name() -> str:
        return "data_mismatch"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        if len(X_gt.columns) != len(X_syn.columns):
            raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_syn.shape}")

        def _eval_score(lhs: pd.Series, rhs: pd.Series) -> int:
            return int(lhs.dtype != rhs.dtype)

        diffs = 0
        for col in X_gt.columns:
            diffs += _eval_score(X_gt[col], X_syn[col])

        return {"score": diffs / (len(X_gt.columns) + 1)}


class CommonRowsProportion(BasicMetricEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_sanity.CommonRowsProportion
        :parts: 1

    Returns the proportion of rows in the real dataset leaked in the synthetic dataset.

    Score:
        0: there are no common rows between the real and synthetic datasets.
        1: all the rows in the real dataset are leaked in the synthetic dataset.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)

    @staticmethod
    def name() -> str:
        return "common_rows_proportion"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        if len(X_gt.columns) != len(X_syn.columns):
            raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_syn.shape}")

        intersection = (
            X_gt.dataframe()
            .merge(X_syn.dataframe(), how="inner", indicator=False)
            .drop_duplicates()
        )
        return {"score": len(intersection) / (len(X_gt) + 1e-8)}


class NearestSyntheticNeighborDistance(BasicMetricEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_sanity.NearestSyntheticNeighborDistance
        :parts: 1

    Computes the <reduction>(distance) from the real data to the closest neighbor in the synthetic data"""

    @staticmethod
    def name() -> str:
        return "nearest_syn_neighbor_distance"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        if len(X_gt.columns) != len(X_syn.columns):
            raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_syn.shape}")

        dist = BasicMetricEvaluator._helper_nearest_neighbor(X_gt, X_syn)

        dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist) + 1e-8)
        return {self._reduction: float(self.reduction()(dist))}


class CloseValuesProbability(BasicMetricEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_sanity.CloseValuesProbability
        :parts: 1

    Compute the probability of close values between the real and synthetic data.

    Score:
        0 means there is no chance to have synthetic rows similar to the real.
        1 means that all the synthetic rows are similar to some real rows.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)

    @staticmethod
    def name() -> str:
        return "close_values_probability"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        if len(X_gt.columns) != len(X_syn.columns):
            raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_syn.shape}")

        dist = BasicMetricEvaluator._helper_nearest_neighbor(X_gt, X_syn)
        dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist) + 1e-8)

        threshold = 0.2

        return {"score": (dist <= threshold).sum() / len(dist)}


class DistantValuesProbability(BasicMetricEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_sanity.DistantValuesProbability
        :parts: 1

    Compute the probability of distant values between the real and synthetic data.

    Score:
        0 means there is no chance to have rows in the synthetic far away from the real data.
        1 means all the synthetic datapoints are far away from the real data.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)

    @staticmethod
    def name() -> str:
        return "distant_values_probability"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        if len(X_gt.columns) != len(X_syn.columns):
            raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_syn.shape}")

        dist = BasicMetricEvaluator._helper_nearest_neighbor(X_gt, X_syn)
        dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist) + 1e-8)

        threshold = 0.8

        return {"score": (dist >= threshold).sum() / len(dist)}
