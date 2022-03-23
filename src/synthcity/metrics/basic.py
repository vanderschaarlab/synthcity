# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.neighbors import NearestNeighbors


def _helper_nearest_neighbor(X_gt: pd.DataFrame, X_synth: pd.DataFrame) -> pd.Series:
    try:
        estimator = NearestNeighbors(n_neighbors=5).fit(X_synth)
        dist, _ = estimator.kneighbors(X_gt, 1, return_distance=True)
        return pd.Series(dist.squeeze(), index=X_gt.index)
    except BaseException:
        return pd.Series([999])


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_data_mismatch_score(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_synth: pd.DataFrame, y_synth: pd.Series
) -> float:
    """Basic sanity score. Compares the data types between the column of the ground truth and the synthetic data.

    Score:
        0: no datatype mismatch.
        1: complete data type mismatch between the datasets.
    """
    if len(X_gt.columns) != len(X_synth.columns):
        raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_synth.shape}")

    def _eval_score(lhs: pd.Series, rhs: pd.Series) -> int:
        return int(lhs.dtype != rhs.dtype)

    diffs = _eval_score(y_gt, y_synth)
    for col in X_gt.columns:
        diffs += _eval_score(X_gt[col], X_synth[col])

    return diffs / (len(X_gt.columns) + 1)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_common_rows_proportion(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_synth: pd.DataFrame, y_synth: pd.Series
) -> float:
    """Returns the proportion of rows in the real dataset leaked in the synthetic dataset.

    Score:
        0: there are no common rows between the real and synthetic datasets.
        1: all the rows in the real dataset are leaked in the synthetic dataset.
    """

    if len(X_gt.columns) != len(X_synth.columns):
        raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_synth.shape}")

    intersection = X_gt.merge(X_synth, how="inner", indicator=False).drop_duplicates()
    return len(intersection) / (len(X_gt) + 1e-8)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_avg_distance_nearest_synth_neighbor(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_synth: pd.DataFrame, y_synth: pd.Series
) -> float:
    """Computes the mean distance from the real data to the closest neighbor in the synthetic data

    Score:
        0: all the real rows are leaked in the synthetic dataset.
        1: all the synthetic rows are far away from the real dataset.
    """

    if len(X_gt.columns) != len(X_synth.columns):
        raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_synth.shape}")

    X_synth["target"] = y_synth
    X_gt["target"] = y_gt

    dist = _helper_nearest_neighbor(X_gt, X_synth)

    dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist) + 1e-8)
    return np.mean(dist)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_inlier_probability(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_synth: pd.DataFrame, y_synth: pd.Series
) -> float:
    """Compute the probability of close values between the real and synthetic data.

    Score:
        0 means there is no chance to have synthetic rows similar to the real.
        1 means that all the synthetic rows are similar to some real rows.
    """

    if len(X_gt.columns) != len(X_synth.columns):
        raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_synth.shape}")

    X_synth["target"] = y_synth
    X_gt["target"] = y_gt

    dist = _helper_nearest_neighbor(X_gt, X_synth)
    dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist) + 1e-8)

    threshold = 0.2

    return (dist <= threshold).sum() / len(dist)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_outlier_probability(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_synth: pd.DataFrame, y_synth: pd.Series
) -> float:
    """Compute the probability of distant values between the real and synthetic data.

    Score:
        0 means there is no chance to have rows in the synthetic far away from the real data.
        1 means all the synthetic datapoints are far away from the real data.
    """

    if len(X_gt.columns) != len(X_synth.columns):
        raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_synth.shape}")

    X_synth["target"] = y_synth
    X_gt["target"] = y_gt

    dist = _helper_nearest_neighbor(X_gt, X_synth)
    dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist) + 1e-8)

    threshold = 0.8

    return (dist >= threshold).sum() / len(dist)
