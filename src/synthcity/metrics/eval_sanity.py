# stdlib
from typing import Any, Dict

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.neighbors import NearestNeighbors

# synthcity absolute
from synthcity.metrics.core import MetricEvaluator


class BasicMetricEvaluator(MetricEvaluator):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def _helper_nearest_neighbor(
        X_gt: pd.DataFrame, X_synth: pd.DataFrame
    ) -> pd.Series:
        try:
            estimator = NearestNeighbors(n_neighbors=5).fit(X_synth)
            dist, _ = estimator.kneighbors(X_gt, 1, return_distance=True)
            return pd.Series(dist.squeeze(), index=X_gt.index)
        except BaseException:
            return pd.Series([999])

    @staticmethod
    def type() -> str:
        return "sanity"


class DataMismatchScore(BasicMetricEvaluator):
    """Basic sanity score. Compares the data types between the column of the ground truth and the synthetic data.

    Score:
        0: no datatype mismatch.
        1: complete data type mismatch between the datasets.
    """

    @staticmethod
    def name() -> str:
        return "data_mismatch"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: pd.DataFrame, X_synth: pd.DataFrame) -> Dict:
        if len(X_gt.columns) != len(X_synth.columns):
            raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_synth.shape}")

        def _eval_score(lhs: pd.Series, rhs: pd.Series) -> int:
            return int(lhs.dtype != rhs.dtype)

        diffs = 0
        for col in X_gt.columns:
            diffs += _eval_score(X_gt[col], X_synth[col])

        return {"score": diffs / (len(X_gt.columns) + 1)}


class CommonRowsProportion(BasicMetricEvaluator):
    """Returns the proportion of rows in the real dataset leaked in the synthetic dataset.

    Score:
        0: there are no common rows between the real and synthetic datasets.
        1: all the rows in the real dataset are leaked in the synthetic dataset.
    """

    @staticmethod
    def name() -> str:
        return "common_rows_proportion"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: pd.DataFrame, X_synth: pd.DataFrame) -> Dict:
        if len(X_gt.columns) != len(X_synth.columns):
            raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_synth.shape}")

        intersection = X_gt.merge(
            X_synth, how="inner", indicator=False
        ).drop_duplicates()
        return {"score": len(intersection) / (len(X_gt) + 1e-8)}


class NearestSyntheticNeighborDistance(BasicMetricEvaluator):
    """Computes the <reduction>(distance) from the real data to the closest neighbor in the synthetic data"""

    @staticmethod
    def name() -> str:
        return "nearest_syn_neighbor_distance"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: pd.DataFrame, X_synth: pd.DataFrame) -> Dict:
        if len(X_gt.columns) != len(X_synth.columns):
            raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_synth.shape}")

        dist = BasicMetricEvaluator._helper_nearest_neighbor(X_gt, X_synth)

        dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist) + 1e-8)
        return {self._reduction: self.reduction()(dist)}


class CloseValuesProbability(BasicMetricEvaluator):
    """Compute the probability of close values between the real and synthetic data.

    Score:
        0 means there is no chance to have synthetic rows similar to the real.
        1 means that all the synthetic rows are similar to some real rows.
    """

    @staticmethod
    def name() -> str:
        return "close_values_probability"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: pd.DataFrame, X_synth: pd.DataFrame) -> Dict:
        if len(X_gt.columns) != len(X_synth.columns):
            raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_synth.shape}")

        dist = BasicMetricEvaluator._helper_nearest_neighbor(X_gt, X_synth)
        dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist) + 1e-8)

        threshold = 0.2

        return {"score": (dist <= threshold).sum() / len(dist)}


class DistantValuesProbability(BasicMetricEvaluator):
    """Compute the probability of distant values between the real and synthetic data.

    Score:
        0 means there is no chance to have rows in the synthetic far away from the real data.
        1 means all the synthetic datapoints are far away from the real data.
    """

    @staticmethod
    def name() -> str:
        return "distant_values_probability"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: pd.DataFrame, X_synth: pd.DataFrame) -> Dict:
        if len(X_gt.columns) != len(X_synth.columns):
            raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_synth.shape}")

        dist = BasicMetricEvaluator._helper_nearest_neighbor(X_gt, X_synth)
        dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist) + 1e-8)

        threshold = 0.8

        return {"score": (dist >= threshold).sum() / len(dist)}
