# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def _encode_scale(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy().fillna(0)

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col])

    X = MinMaxScaler().fit_transform(X)

    return X


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def integrity_score(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_synth: pd.DataFrame, y_synth: pd.Series
) -> float:
    """Basic sanity score. Compares the data types between the column of the ground truth and the synthetic data.

    Lower is better.
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
def avg_common_rows(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_synth: pd.DataFrame, y_synth: pd.Series
) -> float:
    """Returns the proportion of common rows in the ground truth and the synthetic data.

    Lower is better.
    """

    if len(X_gt.columns) != len(X_synth.columns):
        raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_synth.shape}")

    intersection = X_gt.merge(X_synth, how="inner", indicator=False)

    return len(intersection) / (len(X_gt) + 1e-8)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def avg_distance_nearest_synth_neighbor(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_synth: pd.DataFrame, y_synth: pd.Series
) -> float:
    """Returns the mean distance from the real data to the closest neighbor in the synthetic data

    Lower is better.
    """

    if len(X_gt.columns) != len(X_synth.columns):
        raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_synth.shape}")

    X_synth["target"] = y_synth
    X_gt["target"] = y_gt

    X_synth = _encode_scale(X_synth)
    X_gt = _encode_scale(X_gt)

    estimator = NearestNeighbors(n_neighbors=5).fit(X_synth)

    dist, _ = estimator.kneighbors(X_gt, 1, return_distance=True)

    return np.mean(dist)
