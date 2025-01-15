# synthcity/plugins/syn_seq/methods/misc.py

"""
A collection of miscellaneous synthesis methods that do not fit neatly into
the more specific categories (e.g., cart, ctree, logreg, etc.).

Here we provide simple or fallback sampling approaches, such as:

    - syn_random: purely random sampling from the observed y, ignoring X.
    - syn_constant: fill new rows with a constant value.
    - syn_identity: pass-through if the new data shape matches the original. Useful for debugging.
    - syn_swr: "sample without replacement" style, akin to R's "SWR" method in synthpop.

Each function returns a dict with:
    "res" -> the synthesized values for new data Xp
    "fit" -> a dictionary with any fitted objects or metadata (optional)

Usage:
    from .misc import syn_random, syn_constant, syn_identity, syn_swr

    # Example usage:
    # y is observed data (n, ), X is shape (n, p), Xp is shape (m, p)

    result = syn_random(y, X, Xp, random_state=42)
    y_syn = result["res"]

    or

    result = syn_swr(y, X, Xp)
    y_syn = result["res"]
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


def syn_random(
    y: Union[pd.Series, np.ndarray],
    X: Union[pd.DataFrame, np.ndarray],
    Xp: Union[pd.DataFrame, np.ndarray],
    proper: bool = False,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Synthesize new data by simply sampling with replacement from the observed values in y,
    ignoring X or Xp.

    If 'proper=True', we do a bootstrap on (X, y) so that y is resampled before usage,
    aligning with 'proper' imputation concepts.

    Args:
        y: 1D array-like of shape (n,). Observed outcome to sample from.
        X: 2D array-like (n, p). Covariates, not used here but included for signature consistency.
        Xp: 2D array-like (m, p). Covariates for new data, also not used here.
        proper: bool, default=False. If True, bootstrap the training data prior to sampling.
        random_state: int, for reproducibility.
        **kwargs: Unused here, for signature consistency.

    Returns:
        A dictionary with:
            "res": array-like of shape (m,) with the synthetic outcome for each row in Xp.
            "fit": empty dictionary or metadata (optional).
    """
    rng = np.random.RandomState(random_state)
    y_array = y.values if isinstance(y, pd.Series) else np.asarray(y)
    n = len(y_array)

    # If proper => bootstrap (X,y) in the sense that we re-sample y
    if proper:
        idx_boot = rng.choice(n, size=n, replace=True)
        y_array = y_array[idx_boot]

    m = len(Xp) if isinstance(Xp, (pd.DataFrame, np.ndarray)) else 0
    if m == 0:
        raise ValueError("Xp is empty or invalid, cannot generate new samples.")

    # Now sample m times from y_array (with replacement)
    syn_index = rng.randint(low=0, high=len(y_array), size=m)
    syn_values = y_array[syn_index]

    return {
        "res": syn_values,
        "fit": {},
    }


def syn_swr(
    y: Union[pd.Series, np.ndarray],
    X: Union[pd.DataFrame, np.ndarray],
    Xp: Union[pd.DataFrame, np.ndarray],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    'Sample Without Replacement' approach, akin to R's SWR in synthpop:

    - If we want m samples (m = len(Xp)) and m <= len(y):
        we pick a unique random subset of y with no replacement.
    - If m > len(y), then we first pick 'len(y)' unique values from y,
      then continue sampling from y *with replacement* for the remainder.
      This logic mimics the notion that we can't pick more unique values than exist.

    Args:
        y: 1D array-like of shape (n,). Observed outcome to sample from.
        X: 2D array-like (n, p). Covariates for training data (unused here).
        Xp: 2D array-like (m, p). Covariates of new data for which we want synthetic y.
        **kwargs: not used.

    Returns:
        A dictionary with:
            "res": array of shape (m,) with the synthetic y.
            "fit": empty dictionary or metadata.
    """
    rng = np.random.RandomState(kwargs.get("random_state", 0))

    y_array = y.values if isinstance(y, pd.Series) else np.asarray(y)
    n = len(y_array)

    if not isinstance(Xp, (pd.DataFrame, np.ndarray)):
        raise ValueError("Xp is invalid or empty.")

    m = len(Xp)

    if m <= n:
        # pick exactly m unique samples from y
        picks = rng.choice(y_array, size=m, replace=False)
    else:
        # pick n unique samples first
        picks_unique = rng.choice(y_array, size=n, replace=False)
        # then sample the difference with replacement
        overshoot = m - n
        picks_extra = rng.choice(y_array, size=overshoot, replace=True)
        picks = np.concatenate([picks_unique, picks_extra])

    return {
        "res": picks,
        "fit": {},
    }


def syn_constant(
    y: Union[pd.Series, np.ndarray],
    X: Union[pd.DataFrame, np.ndarray],
    Xp: Union[pd.DataFrame, np.ndarray],
    const_value: Any = 0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Synthesize data by returning a constant value for each new record, ignoring y or X.

    Useful as a placeholder or sentinel fill.

    Args:
        y: 1D array-like, shape (n,). Unused, included for signature consistency.
        X: 2D array-like, shape (n, p). Unused.
        Xp: 2D array-like, shape (m, p). The size (m) determines how many values to output.
        const_value: The constant to fill.
        **kwargs: unused.

    Returns:
        {
            "res": shape (m,) array of the constant,
            "fit": {}
        }
    """
    if isinstance(Xp, (pd.DataFrame, np.ndarray)):
        m = len(Xp)
    else:
        raise ValueError("Xp is invalid or empty")

    syn_values = np.full(m, const_value, dtype=object)

    return {
        "res": syn_values,
        "fit": {},
    }


def syn_identity(
    y: Union[pd.Series, np.ndarray],
    X: Union[pd.DataFrame, np.ndarray],
    Xp: Union[pd.DataFrame, np.ndarray],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    A trivial pass-through approach:
      - If len(Xp) == len(y), we just return y as is.
      - Otherwise, raise an error.

    Useful for debugging or chain-of-thought checks.

    Args:
        y: 1D array-like of shape (n,).
        X: 2D array-like (n, p). Unused.
        Xp: 2D array-like (m, p). If m != n, raises error.
        **kwargs: Unused.

    Returns:
        {
            "res": simply y if lengths match
            "fit": {}
        }
    """
    n = len(y)
    if not isinstance(Xp, (pd.DataFrame, np.ndarray)):
        raise ValueError("Xp is invalid or empty")

    m = len(Xp)
    if m != n:
        raise ValueError(
            f"syn_identity requires len(Xp) == len(y). Got y={n}, Xp={m}."
        )

    y_array = y.values if isinstance(y, pd.Series) else np.asarray(y)

    return {
        "res": y_array.copy(),  # or just y_array
        "fit": {},
    }
