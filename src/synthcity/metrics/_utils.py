# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def encode_scale(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy().fillna(0)

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col])

    X = MinMaxScaler().fit_transform(X)

    return X


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_freq(X_gt: pd.DataFrame, X_synth: pd.DataFrame) -> dict:
    """Get percentual frequencies for each possible real categorical value.

    Returns:
        The observed and expected frequencies (as a percent).
    """
    res = {}
    for col in X_gt.columns:
        if len(X_gt[col].unique()) < 5:  # categorical
            gt = (X_gt[col].value_counts() / len(X_gt)).to_dict()
            synth = (X_synth[col].value_counts() / len(X_synth)).to_dict()
        else:
            gt_vals, bins = np.histogram(X_gt[col])
            synth_vals, _ = np.histogram(X_synth[col], bins=bins)
            gt = {k: v / (sum(gt_vals) + 1e-8) for k, v in zip(bins, gt_vals)}
            synth = {k: v / (sum(synth_vals) + 1e-8) for k, v in zip(bins, synth_vals)}

        for val in gt:
            if val not in synth or synth[val] == 0:
                synth[val] = 1e-11
        for val in synth:
            if val not in gt or gt[val] == 0:
                gt[val] = 1e-11

        assert gt.keys() == synth.keys()
        res[col] = (list(gt.values()), list(synth.values()))

    return res
