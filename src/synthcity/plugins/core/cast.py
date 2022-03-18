# stdlib
from typing import Any

# third party
import numpy as np
import pandas as pd


def to_dataframe(X: Any) -> pd.DataFrame:
    """Helper for casting arguments to `pandas.DataFrame`.

    Args:
        X: the object to cast.

    Returns:
        pd.DataFrame: the converted DataFrame.

    Raises:
        ValueError: if the argument cannot be converted to a DataFrame.
    """
    if isinstance(X, pd.DataFrame):
        return X
    elif isinstance(X, (list, np.ndarray, pd.core.series.Series)):
        return pd.DataFrame(X)

    raise ValueError(
        f"unsupported data type {type(X)}. Try list, pandas.DataFrame or numpy.ndarray"
    )


def to_ndarray(X: Any) -> np.ndarray:
    """Helper for casting arguments to `numpy.ndarray`.

    Args:
        X: the object to cast.

    Returns:
        pd.DataFrame: the converted ndarray.

    Raises:
        ValueError: if the argument cannot be converted to a ndarray.
    """
    if isinstance(X, np.ndarray):
        return X
    elif isinstance(X, (list, pd.DataFrame, pd.core.series.Series)):
        return np.array(X)

    raise ValueError(
        f"unsupported data type {type(X)}. Try list, pandas.DataFrame or numpy.ndarray"
    )


__all__ = [
    "to_dataframe",
    "to_ndarray",
]
