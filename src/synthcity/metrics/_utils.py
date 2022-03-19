# stdlib
from typing import Tuple

# third party
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
def get_freq(X_gt: pd.DataFrame, X_synth: pd.DataFrame) -> Tuple:
    """Get percentual frequencies for each possible real categorical value.

    Returns:
        The observed and expected frequencies (as a percent).
    """
    pass
