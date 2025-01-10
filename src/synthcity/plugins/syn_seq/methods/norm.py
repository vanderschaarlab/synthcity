# synthcity/plugins/syn_seq/methods/norm.py

"""
This module provides "normal regression style" synthesis methods inspired by the
R `synthpop` package's approach (syn.norm, syn.lognorm, syn.sqrtnorm, etc.).
These methods handle numeric data transformations (log, sqrt, etc.) and
perform regression-based synthesis with either fixed or Bayesian draws.

NOTE: This version has been slightly refactored for clarity while preserving
the overall logic and flow of the original reference. Key functions include:
- syn_norm
- syn_lognorm
- syn_sqrtnorm
- syn_cubertnorm
- syn_normrank
- syn_ranknorm

All are designed to fit a normal (linear) model (or transform -> linear model -> inverse transform),
and then generate synthetic data accordingly.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


def _decimal_places(values: np.ndarray) -> int:
    """
    Approximate the maximum number of decimal places among the given numeric values.
    In R code, decimalplaces() tries to find how many decimals are needed to avoid
    losing information when rounding. We do a simplified check.
    """
    if len(values) == 0:
        return 0

    valid = values[~pd.isna(values)]
    if len(valid) == 0:
        return 0

    max_decimals = 0
    for val in valid:
        s = f"{val:.16g}"  # 16-digit precision
        if "." in s:
            dec_len = len(s.split(".")[1].rstrip("0"))
            max_decimals = max(max_decimals, dec_len)
    return max_decimals


def _norm_fix_syn(y: np.ndarray, X: np.ndarray, ridge: float = 1e-5) -> Dict[str, Any]:
    """
    Fit a linear model: y ~ X, using a small ridge penalty.
    Return fitted coefficients and residual sigma.

    Returns:
        {
            "beta": array of shape (p, ),
            "sigma": float,
        }
    """
    xtx = X.T @ X
    pen = ridge * np.diag(np.diag(xtx))
    v_inv = np.linalg.inv(xtx + pen)

    beta_hat = v_inv @ X.T @ y
    residuals = y - X @ beta_hat
    dof = max(len(y) - X.shape[1] - 1, 1)
    sigma = np.sqrt(np.sum(residuals ** 2) / dof)

    return {"beta": beta_hat, "sigma": sigma}


def _norm_draw_syn(y: np.ndarray, X: np.ndarray, ridge: float = 1e-5) -> Dict[str, Any]:
    """
    Bayesian version of the linear model for normal data:
      1) Fit the fixed regression => (coef, sigma)
      2) Draw sigma_star from scaled inverse-chi-square
      3) Draw beta_star ~ Normal( beta_hat, sigma_star^2 * V )

    Where V = (X^T X + ridgeI)^-1.
    """
    # 1) fixed fit
    fixed_pars = _norm_fix_syn(y, X, ridge=ridge)
    beta_hat = fixed_pars["beta"]
    sigma_hat = fixed_pars["sigma"]

    xtx = X.T @ X
    pen = ridge * np.diag(np.diag(xtx))
    v_inv = np.linalg.inv(xtx + pen)

    # 2) draw sigma_star
    dof = max(len(y) - X.shape[1], 1)
    chi = np.random.chisquare(dof)
    sigma_star = np.sqrt((len(y) - X.shape[1]) * (sigma_hat ** 2) / chi)

    # 3) draw beta_star
    L = np.linalg.cholesky((v_inv + v_inv.T) / 2.0)
    beta_star = beta_hat + (L @ np.random.normal(size=len(beta_hat))) * sigma_star

    return {"coef": beta_hat, "beta": beta_star, "sigma": sigma_star}


def syn_norm(
    y: np.ndarray,
    X: np.ndarray,
    Xp: np.ndarray,
    proper: bool = False,
    ridge: float = 1e-5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Synthesis of y given X using a normal linear regression, possibly with Bayesian draws.
    This is analogous to R's syn.norm.

    Steps:
      1) Fit linear model y~X (with or without Bayesian draws).
      2) Predict new rows Xp and add random noise ~ N(0, sigma^2).
      3) Round the output according to decimal places from the original y.

    Args:
        y: (n,) array of numeric response.
        X: (n, p) matrix of predictors.
        Xp: (m, p) matrix for generating new values.
        proper: if True, do Bayesian draws (._norm_draw_syn), else fixed parameters.
        ridge: small penalty factor for numerical stability.
        **kwargs: unused extras for compatibility.

    Returns:
        {
            "res": (m,) array of synthetic y,
            "fit": dict with parameters used or drawn,
        }
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    Xp = np.asarray(Xp, dtype=float)

    if not proper:
        fitpars = _norm_fix_syn(y, X, ridge=ridge)
        beta = fitpars["beta"]
        sigma = fitpars["sigma"]
    else:
        fitpars = _norm_draw_syn(y, X, ridge=ridge)
        beta = fitpars["beta"]
        sigma = fitpars["sigma"]

    mean_pred = Xp @ beta
    noise = np.random.normal(0, sigma, size=len(mean_pred))
    synthetic = mean_pred + noise

    nd = _decimal_places(y)
    synthetic = np.round(synthetic, nd)

    return {"res": synthetic, "fit": fitpars}


def syn_lognorm(
    y: np.ndarray,
    X: np.ndarray,
    Xp: np.ndarray,
    proper: bool = False,
    ridge: float = 1e-5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Synthesis for log(y) -> normal. Then exponentiate to get final.

    Similar to R's syn.lognorm:
      1) y -> offset if needed (avoid log(0)).
      2) ylog = log(y + offset)
      3) do syn_norm
      4) invert via exp()

    Args:
        y: Original values (must be >= 0).
        X: training predictors.
        Xp: new rows to generate from.
        proper: if True, do Bayesian draws.
        ridge: penalty factor.

    Returns:
        {
          "res": exponentiated synthetic,
          "fit": parameters used
        }
    """
    y = np.asarray(y, dtype=float)
    if np.any(y < 0):
        raise ValueError("Cannot log-transform negative y in syn_lognorm.")

    offset_val = 0.0
    if np.any(y == 0):
        # shift by half of the smallest positive
        positive_vals = y[y > 0]
        if len(positive_vals) == 0:
            raise ValueError("All y=0? Cannot do log transform.")
        offset_val = 0.5 * positive_vals.min()
    y_shifted = y + offset_val
    ylog = np.log(y_shifted)

    # do normal
    res_dict = syn_norm(ylog, X, Xp, proper=proper, ridge=ridge, **kwargs)
    log_syn = res_dict["res"]

    # revert
    exp_vals = np.exp(log_syn) - offset_val
    # protect from negatives if offset is large
    exp_vals = np.where(exp_vals < 0, 0, exp_vals)

    original_nd = _decimal_places(y)
    exp_vals = np.round(exp_vals, original_nd)

    res_dict["res"] = exp_vals
    return res_dict


def syn_sqrtnorm(
    y: np.ndarray,
    X: np.ndarray,
    Xp: np.ndarray,
    proper: bool = False,
    ridge: float = 1e-5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Synthesis for sqrt(y):
        1) y -> sqrt(y)
        2) do syn_norm
        3) invert => square

    Args:
        y: must be >= 0
    """
    y = np.asarray(y, dtype=float)
    if np.any(y < 0):
        raise ValueError("Cannot sqrt negative y in syn_sqrtnorm.")

    y_sqrt = np.sqrt(y)
    res_dict = syn_norm(y_sqrt, X, Xp, proper=proper, ridge=ridge, **kwargs)
    pred_sqrt = res_dict["res"]
    pred = np.power(pred_sqrt, 2)

    nd = _decimal_places(y)
    res_dict["res"] = np.round(pred, nd)
    return res_dict


def syn_cubertnorm(
    y: np.ndarray,
    X: np.ndarray,
    Xp: np.ndarray,
    proper: bool = False,
    ridge: float = 1e-5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Synthesis for y^(1/3):
        1) y -> y^(1/3)
        2) syn_norm
        3) invert => cube

    Args:
        y: must be >= 0
    """
    y = np.asarray(y, dtype=float)
    if np.any(y < 0):
        raise ValueError("Cannot cuberoot negative values in syn_cubertnorm.")

    y_cubert = np.power(y, 1.0 / 3.0)
    res_dict = syn_norm(y_cubert, X, Xp, proper=proper, ridge=ridge, **kwargs)
    pred_cubert = res_dict["res"]
    pred = np.power(pred_cubert, 3)

    nd = _decimal_places(y)
    res_dict["res"] = np.round(pred, nd)
    return res_dict


def _syn_smooth(ysyn: np.ndarray, yobs: np.ndarray) -> np.ndarray:
    """
    A naive "density smoothing" approximation:
      1) estimate a small bandwidth from yobs
      2) add random N(0,bw) to the synthetic values
      3) clamp to [min(yobs), max(yobs)]
      4) round to the same decimals as yobs
    """
    import warnings

    valid = ysyn[~pd.isna(ysyn)]
    if len(valid) < 2:
        return ysyn

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bw = np.std(valid) * 0.1
    if bw <= 0:
        return ysyn

    addvals = np.random.normal(loc=0.0, scale=bw, size=len(valid))
    newvals = valid + addvals

    mn, mx = np.min(yobs), np.max(yobs)
    newvals = np.clip(newvals, mn, mx)

    dec = _decimal_places(yobs)
    newvals = np.round(newvals, dec)

    ysyn[~pd.isna(ysyn)] = newvals
    return ysyn


try:
    from math import erfinv
except ImportError:
    from scipy.special import erfinv

def syn_normrank(
    y: np.ndarray,
    X: np.ndarray,
    Xp: np.ndarray,
    smoothing: str = "",
    proper: bool = False,
    ridge: float = 1e-5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    'Rank-based' normal approach:
      1) Convert y -> rank -> z-scores => syn_norm => predicted z => transform back to rank => pick from y distribution
      2) optional smoothing

    This is analogous to R's syn.normrank logic.
    """
    from scipy.stats import norm

    y = np.asarray(y, dtype=float)
    n = len(y)

    # rank transform
    ranks = np.argsort(np.argsort(y))  # integer ranks from 0..n-1
    rank_prob = (ranks + 1) / (n + 1.0)
    z = np.sqrt(2) * erfinv(2 * rank_prob - 1)

    # syn_norm on z
    res_dict = syn_norm(z, X, Xp, proper=proper, ridge=ridge, **kwargs)
    z_pred = res_dict["res"]

    # map z_pred to new ranks
    cdf_vals = norm.cdf(z_pred)
    new_ranks_float = cdf_vals * (n + 1)
    new_ranks_float[new_ranks_float < 1] = 1
    new_ranks_float[new_ranks_float > n] = n
    new_ranks = new_ranks_float.astype(int)

    sorted_y = np.sort(y)
    syn_y = sorted_y[new_ranks - 1]

    # optional smoothing
    if smoothing == "density":
        syn_y = _syn_smooth(syn_y, y)

    res_dict["res"] = syn_y
    return res_dict


def syn_ranknorm(
    y: np.ndarray,
    X: np.ndarray,
    Xp: np.ndarray,
    proper: bool = False,
    ridge: float = 1e-5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Another rank-based approach:
      - Fit normal on raw y => predict => rank predicted => sample from y distribution by rank
    """
    n = len(y)
    y = np.asarray(y, dtype=float)

    # standard syn_norm on y
    res_dict = syn_norm(y, X, Xp, proper=proper, ridge=ridge, **kwargs)
    pred_y = res_dict["res"]

    # rank transform predicted
    rank_pred = np.argsort(np.argsort(pred_y))
    # create a bootstrap sample from y
    y_boot = np.random.choice(y, size=n, replace=True)
    y_boot_sorted = np.sort(y_boot)

    final_y = y_boot_sorted[rank_pred]
    res_dict["res"] = final_y
    return res_dict
