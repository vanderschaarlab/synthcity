# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.neighbors import NearestNeighbors


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def integrity_score(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_synth: pd.DataFrame, y_synth: pd.Series
) -> int:
    """Basic sanity score. Compares the data types between the column of the ground truth and the synthetic data.

    Lower is better.
    """
    if len(X_gt.columns) != len(X_synth.columns):
        raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_synth.shape}")

    def _eval_score(lhs: pd.Series, rhs: pd.Series):
        return int(lhs.dtype != rhs.dtype)

    diffs = _eval_score(y_gt, y_synth)
    for col in X_gt.columns:
        diffs += _eval_score(X_gt[col], X_synth[col])

    return diffs / (len(X_gt.columns) + 1)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def common_rows(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_synth: pd.DataFrame, y_synth: pd.Series
) -> int:
    """Returns the number of common rows in the ground truth and the synthetic data.

    Lower is better.
    """

    if len(X_gt.columns) != len(X_synth.columns):
        raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_synth.shape}")

    intersection = X_gt.merge(X_synth, how="inner", indicator=False)

    return len(intersection) / (len(X_gt) + 1e-8)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def nearest_synth_neighbor(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_synth: pd.DataFrame, y_synth: pd.Series
) -> int:
    """ """

    if len(X_gt.columns) != len(X_synth.columns):
        raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_synth.shape}")

    X_synth["target"] = y_synth
    X_gt["target"] = y_gt

    estimator = NearestNeighbors(n_neighbors=5).fit(X_synth)

    dist, _ = estimator.kneighbors(X_gt, 1, return_distance=True)

    return np.mean(dist)
