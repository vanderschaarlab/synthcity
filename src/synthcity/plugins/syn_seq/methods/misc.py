# synthcity/plugins/syn_seq/methods/misc.py

"""
A collection of miscellaneous synthesis methods that do not fit neatly into
the more specific categories (e.g., cart, ctree, logreg, etc.).

For example, we might include:
    - syn_random: purely random sampling from observed values
    - syn_constant: fill with a constant value
    - syn_identity: pass-through without changes (debug/trivial approach)
    - etc.

Each function returns a dict with:
    "res" -> the synthesized values for new data Xp
    "fit" -> a dictionary with any fitted objects or metadata (optional)

Usage:
    from misc import syn_random

    result = syn_random(y, X, Xp, random_state=42)
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
    Synthesize new data by simply sampling from the observed values in y,
    ignoring X or Xp.

    If 'proper=True', we apply a bootstrap on (X, y) to keep it consistent
    with the idea of 'proper' imputation.

    Args:
        y: 1D array of shape (n,). Observed outcome to sample from.
        X: 2D array (n, p). Covariates, not used in this method but included for signature consistency.
        Xp: 2D array (m, p). Covariates for new data, also not used here.
        proper: bool, default=False. If True, bootstrap the training data prior to sampling.
        random_state: Optional int. For reproducibility.
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

    # Now sample m times from y_array
    syn_index = rng.randint(low=0, high=len(y_array), size=m)
    syn_values = y_array[syn_index]

    return {
        "res": syn_values,
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
    Synthesize data by returning a constant value for each new record, ignoring the training data.

    Example use-case: Fill with a placeholder or sentinel value.

    Args:
        y: 1D array of shape (n,). Unused here, but included for signature consistency.
        X: 2D array of shape (n, p). Unused.
        Xp: 2D array of shape (m, p). Only used to determine how many outputs needed.
        const_value: The constant to return for each new record.
        **kwargs: Unused.

    Returns:
        A dictionary with:
            "res": array-like of shape (m,) with the constant outcome for each row in Xp.
            "fit": empty dictionary or metadata (optional).
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
    A trivial identity-based approach: if the new data Xp has the same length as y,
    we just return y as is. Otherwise, raises an error. This can be useful for debugging.

    Args:
        y: 1D array of shape (n,).
        X: 2D array of shape (n, p). Unused.
        Xp: 2D array of shape (m, p). If m != n, raises error.
        **kwargs: Unused.

    Returns:
        A dictionary with:
            "res": simply y (if lengths match).
            "fit": empty dictionary or metadata.
    """
    n = len(y)
    if isinstance(Xp, (pd.DataFrame, np.ndarray)):
        m = len(Xp)
    else:
        raise ValueError("Xp is invalid or empty")

    if n != m:
        raise ValueError(
            f"syn_identity requires that len(Xp) == len(y). Got y={n}, Xp={m}."
        )
    y_array = y.values if isinstance(y, pd.Series) else np.asarray(y)

    return {
        "res": y_array.copy(),  # or just y_array
        "fit": {},
    }


# If we want to unify them under a single approach:
# e.g. def syn_misc(method="random"/"constant"/"identity", ...):
# but for clarity, we keep them as separate methods.
