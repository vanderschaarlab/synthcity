# synthcity/plugins/syn_seq/methods/pmm.py

"""
Implements a Predictive Mean Matching (PMM) approach, analogous to R's syn.pmm from the synthpop package.

Key PMM ideas from R syn.pmm:
    - Fit a linear model on the observed data (X -> y).
    - Obtain predicted values yhat for both the observed data (training set) and the new data Xp.
    - For each new row (in Xp), find the 'donors' in the training set whose predicted yhat_obs is closest to yhat_mis.
    - Randomly sample from those donors' original observed y to get the synthetic values.
    - Optionally do "proper" synthesis by bootstrapping (resampling) before fitting the model, as in R.

Here, we provide a simple normal linear regression fit (like syn.norm but 'fixed') by default.

Usage example:
    result = syn_pmm(
        y, X, Xp,
        donors=3,
        proper=False,
        random_state=42,
        ...
    )
    y_syn = result["res"]
    model_info = result["fit"]
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _norm_fix_syn(
    y: np.ndarray,
    X: np.ndarray,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Equivalent to a simple linear regression fit for a 'fixed' approach.
    Returns a dictionary with:
        "coef": the fitted coefficients (1D array)
        "intercept": the fitted intercept (float)
        "model": the fitted LinearRegression model (for debugging/reuse)
    """
    model = LinearRegression(**kwargs)
    model.fit(X, y)
    return {
        "coef": model.coef_,
        "intercept": model.intercept_,
        "model": model,
    }


def _pmm_match_single(
    z_value: float,
    yhat_obs: np.ndarray,
    y_obs: np.ndarray,
    donors: int,
    rng: np.random.RandomState,
) -> float:
    """
    For a single predicted value z_value, find the `donors` closest yhat_obs,
    then randomly pick one from those donors' observed y as the synthetic value.

    Args:
        z_value: float. The predicted value for the row we want to synthesize.
        yhat_obs: array of shape (n,). Predicted values for the original data.
        y_obs: array of shape (n,). Actual observed y values in the training set.
        donors: int. Number of nearest donors to sample from.
        rng: a np.random.RandomState for reproducibility.

    Returns:
        A float value from the donor pool (observed y values).
    """
    # Distances to each predicted yhat_obs
    diffs = np.abs(yhat_obs - z_value)
    # Indices of the top donors
    top_indices = np.argsort(diffs)[:donors]
    # Randomly pick one from these donors
    chosen_idx = rng.choice(top_indices, size=1)[0]
    return y_obs[chosen_idx]


def syn_pmm(
    y: np.ndarray,
    X: np.ndarray,
    Xp: np.ndarray,
    donors: int = 3,
    proper: bool = False,
    random_state: Optional[int] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Predictive Mean Matching for numeric y.

    Args:
        y: (n,) array of numeric response values.
        X: (n, p) array of training covariates.
        Xp: (m, p) array for new data, the rows to synthesize.
        donors: how many nearest donors to sample from for each new row.
        proper: bool. If True, apply bootstrap to (X, y) before fitting (proper imputation).
        random_state: optional integer seed for reproducibility.
        **kwargs: Additional arguments for the linear model.

    Returns:
        A dictionary with:
          "res": (m,) array of synthetic y values for each row in Xp.
          "fit": dict with "coef", "intercept", "model" describing the fitted model.
    """
    # Ensure correct type
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    Xp = np.asarray(Xp, dtype=float)

    rng = np.random.RandomState(random_state)

    # If proper => bootstrap the training data
    if proper:
        n = len(y)
        idx_boot = rng.choice(n, size=n, replace=True)
        X = X[idx_boot, :]
        y = y[idx_boot]

    # Fit a simple linear regression
    fit_info = _norm_fix_syn(y, X, **kwargs)
    beta = fit_info["coef"]
    intercept = fit_info["intercept"]

    # Predicted values for the observed data
    yhat_obs = X @ beta + intercept
    # Predicted values for the new data
    yhat_mis = Xp @ beta + intercept

    # Match and sample
    synthetic_vals = []
    for z_val in yhat_mis:
        picked_val = _pmm_match_single(
            z_value=z_val,
            yhat_obs=yhat_obs,
            y_obs=y,
            donors=donors,
            rng=rng,
        )
        synthetic_vals.append(picked_val)
    synthetic_vals = np.array(synthetic_vals)

    return {
        "res": synthetic_vals,
        "fit": fit_info,
    }
