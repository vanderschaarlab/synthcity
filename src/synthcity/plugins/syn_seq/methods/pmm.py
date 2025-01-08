# synthcity/plugins/syn_seq/methods/pmm.py

"""
Implements a Predictive Mean Matching (PMM) approach, analogous to R's syn.pmm from the synthpop package.

Key ideas from R syn.pmm:
    - Fit a linear model (or similar) on the observed data (X -> y).
    - Obtain predicted values yhat for both the observed data (training set) and the new Xp data.
    - For each row in Xp, find the 'donors' in the training set whose predicted yhatobs is closest to yhatmis.
    - Randomly sample from those donors' original observed y to get the synthetic values.
    - Optionally do "proper" synthesis by bootstrapping (resampling) before fitting the model, as in R.

In this implementation:
    - We provide a simple normal linear regression fit (like .norm.fix.syn from R) by default.
    - The default number of donors is 3, but can be changed with `donors`.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _norm_fix_syn(
    y: np.ndarray, X: np.ndarray, **kwargs: Any
) -> Dict[str, Any]:
    """
    Equivalent to a simple linear regression fit for the 'fixed' approach (not drawing from posterior).
    Returns a dictionary with:
        "coef": the fitted coefficients
        "intercept": the intercept
    """
    model = LinearRegression(**kwargs)
    model.fit(X, y)
    return {
        "coef": model.coef_,
        "intercept": model.intercept_,
        # store the model if we want to debug or retrieve more info
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
    and pick one random value from the corresponding observed y.
    """
    # 1. distances to each yhat_obs
    diffs = np.abs(yhat_obs - z_value)
    # 2. find indices of the top donors
    top_indices = np.argsort(diffs)[:donors]
    # 3. random pick from those donors
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
    Synthesis of numeric y by predictive mean matching.

    Args:
        y: (n,) array of numeric response values.
        X: (n, p) array for the training data's covariates.
        Xp: (m, p) array for the new data (the rows we want to synthesize).
        donors: how many nearest donors to sample from in the original data for each new row.
        proper: bool. If True, do a bootstrap sample of the original data (X, y) before fitting.
        random_state: random seed for reproducibility.
        **kwargs: Additional arguments for the linear regression or for advanced usage.

    Returns:
        {
          "res": array of shape (m,), the synthetic y for each row in Xp
          "fit": a dictionary describing the fitted model and related info
        }
    """
    # Convert to numpy arrays if not already
    y = np.array(y, dtype=float).ravel()
    X = np.array(X, dtype=float)
    Xp = np.array(Xp, dtype=float)

    rng = np.random.RandomState(random_state)

    # If proper, bootstrap the data first
    if proper:
        n = len(y)
        idx_boot = rng.choice(n, size=n, replace=True)
        X = X[idx_boot, :]
        y = y[idx_boot]

    # Fit the linear model
    fit_info = _norm_fix_syn(y, X, **kwargs)
    beta = fit_info["coef"]
    intercept = fit_info["intercept"]

    # Predicted values for observed data
    yhat_obs = X @ beta + intercept

    # Predicted values for new data
    yhat_mis = Xp @ beta + intercept

    # For each row in Xp, do the PMM match
    res_synth = []
    for z_val in yhat_mis:
        sampled_val = _pmm_match_single(
            z_value=z_val,
            yhat_obs=yhat_obs,
            y_obs=y,
            donors=donors,
            rng=rng,
        )
        res_synth.append(sampled_val)
    res_synth = np.array(res_synth)

    return {
        "res": res_synth,
        "fit": fit_info,
    }
