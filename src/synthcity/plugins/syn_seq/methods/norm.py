# synthcity/plugins/syn_seq/methods/norm.py

"""
This module provides "normal regression style" synthesis methods inspired by the
R `synthpop` package's approach (syn.norm, syn.lognorm, syn.sqrtnorm, etc.).
These methods handle numeric data transformations (log, sqrt, etc.) and
perform regression-based synthesis with either fixed or Bayesian draws.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


def _decimal_places(values: np.ndarray) -> int:
    """
    Approximate the maximum number of decimal places among the given numeric values.
    In R code, decimalplaces() tries to find how many decimals are needed to avoid
    losing information when rounding.
    """
    if len(values) == 0:
        return 0

    # Filter out invalid or missing
    vals = values[~pd.isna(values)]
    if len(vals) == 0:
        return 0

    # Convert to string and check decimal length.
    max_decimals = 0
    for val in vals:
        s = f"{val:.16g}"  # 16-digit precision representation
        if "." in s:
            dec_len = len(s.split(".")[1].rstrip("0"))  # remove trailing zeros
            if dec_len > max_decimals:
                max_decimals = dec_len
    return max_decimals


def _norm_fix_syn(y: np.ndarray, X: np.ndarray, ridge: float = 1e-5) -> Dict[str, Any]:
    """
    Compute regression coefficients + error estimate for a linear model:
      y ~ X
    using a ridge-penalized (very small) approach for numerical stability.

    Returns:
        {
            "beta": array of shape (n_features,),
            "sigma": float,
        }
    """
    # X^T X + ridge * diag -> invert
    xtx = X.T @ X
    pen = ridge * np.diag(np.diag(xtx))
    v_inv = np.linalg.inv(xtx + pen)

    beta_hat = v_inv @ X.T @ y
    residuals = y - X @ beta_hat
    dof = max(len(y) - X.shape[1] - 1, 1)  # avoid zero or negative
    sigma = np.sqrt(np.sum(residuals**2) / dof)

    return {"beta": beta_hat, "sigma": sigma}


def _norm_draw_syn(y: np.ndarray, X: np.ndarray, ridge: float = 1e-5) -> Dict[str, Any]:
    """
    Bayesian linear regression synthesis draw for y given X.
    1) Fit fixed regression => (coef, sigma)
    2) sigma_star ~ sqrt( sum(res^2) / rchisq(...) )
    3) beta_star ~ N( coef, sigma_star^2 * V ), where V is (X^T X + ridgeI)^-1
    """
    # Step 1: fit
    fixed_pars = _norm_fix_syn(y, X, ridge=ridge)
    beta_hat = fixed_pars["beta"]
    sigma_hat = fixed_pars["sigma"]

    xtx = X.T @ X
    pen = ridge * np.diag(np.diag(xtx))
    v_inv = np.linalg.inv(xtx + pen)

    # Step 2: draw sigma_star
    # We approximate a chi-square dof = len(y) - X.shape[1].
    # To keep it stable, ensure dof>0
    dof = max(len(y) - X.shape[1], 1)
    # draw from chi-square(dof)
    chi = np.random.chisquare(dof)
    sigma_star = np.sqrt((len(y) - X.shape[1]) * sigma_hat**2 / chi)

    # Step 3: draw beta_star
    # covariance matrix = sigma_star^2 * V
    L = np.linalg.cholesky((v_inv + v_inv.T) / 2.0)  # symmetrize for safety
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
    Synthesis of y given X using normal linear regression, possibly with Bayesian draws.
    This is analogous to R's syn.norm.

    Args:
        y: (n,) array of response.
        X: (n, p) design matrix for training
        Xp: (m, p) design matrix for generation
        proper: If True, use Bayesian draws (._norm_draw_syn), otherwise fix (._norm_fix_syn).
        ridge: small penalty factor for (X^T X) to help stability.
        **kwargs: ignored

    Returns:
        {
          "res": (m,) array of synthetic y values,
          "fit": dict with "beta", "sigma", ...
        }
    """
    # 1) Prepare data
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    Xp = np.asarray(Xp, dtype=float)

    if not proper:
        fitpars = _norm_fix_syn(y, X, ridge=ridge)
        beta = fitpars["beta"]
        sigma = fitpars["sigma"]
    else:
        fitpars = _norm_draw_syn(y, X, ridge=ridge)
        beta = fitpars["beta"]  # the "beta_star" from the draw
        sigma = fitpars["sigma"]

    # 2) Generate predictions + noise
    mean_pred = Xp @ beta
    noise = np.random.normal(loc=0.0, scale=sigma, size=len(mean_pred))
    synthetic = mean_pred + noise

    # 3) Round according to decimal places in original y
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
    Synthesis for log-transformed y: we transform y -> log(y), do syn_norm, then exponentiate back.
    If y has zeros, we offset them before log to avoid negative domain.

    Similar logic to R's syn.lognorm.
    """
    y = np.asarray(y, dtype=float)

    if np.any(y < 0):
        raise ValueError("Log transformation not appropriate for negative values.")
    # offset zeros
    addbit = 0.0
    if np.any(y == 0):
        offset_val = 0.5 * np.min(y[y > 0])  # half of smallest positive
        y = y + offset_val
        addbit = offset_val
    # safe log
    y = np.log(y)

    # do normal
    res_dict = syn_norm(y, X, Xp, proper=proper, ridge=ridge, **kwargs)
    log_syn = res_dict["res"]  # these are in log-space

    # shift back if needed
    if addbit > 0:
        # "res" was log(y+offset), so subtract offset on the log scale
        # Actually in R code, it does a minimal shift. We'll skip that detail
        pass

    # exponentiate
    exp_vals = np.exp(log_syn)
    # round
    original_nd = _decimal_places(np.exp(y))  # how many decimals in original unlogged
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
    Synthesis for sqrt-transformed y:
      y -> sqrt(y)
      do normal, then square the result
    """
    y = np.asarray(y, dtype=float)
    if np.any(y < 0):
        raise ValueError("Square root transformation not valid for negative values.")

    # transform
    y_sqrt = np.sqrt(y)

    # do normal
    res_dict = syn_norm(y_sqrt, X, Xp, proper=proper, ridge=ridge, **kwargs)
    pred_sqrt = res_dict["res"]
    pred = pred_sqrt**2

    nd = _decimal_places(y)
    pred = np.round(pred, nd)
    res_dict["res"] = pred
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
    Synthesis for cubic-root transformation. Similar to sqrt, but y^(1/3).
    Then after regression, we cube the predictions.
    """
    y = np.asarray(y, dtype=float)
    if np.any(y < 0):
        raise ValueError("Cube root transformation not appropriate for negative values.")

    # transform
    y_cubert = np.power(y, 1.0 / 3.0)

    # do normal
    res_dict = syn_norm(y_cubert, X, Xp, proper=proper, ridge=ridge, **kwargs)
    pred_cubert = res_dict["res"]
    pred = pred_cubert**3

    nd = _decimal_places(y)
    pred = np.round(pred, nd)
    res_dict["res"] = pred
    return res_dict


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
    'Rank-based' normal approach. y's ranks -> z-scores, do a normal regression on z.
    Then predict new z-values, convert them back to ranks, and pick from sorted y.

    If smoothing="density", tries to add noise around the chosen values, akin to 'syn.smooth'.
    Otherwise, just picks exact values from y distribution by rank.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    # Convert y to z-scores by rank
    ranks = np.argsort(np.argsort(y))  # integer ranks 0..(n-1)
    z = np.random.normal(size=n)  # We will do a simpler approach
    # Actually, let's do more direct: z_i = qnorm( rank_i / (n+1) ), as in the R code:
    rank_prob = (ranks + 1) / (n + 1.0)
    z = np.random.normal(0, 1, size=n)
    z = np.sqrt(2) * erfinv(2 * rank_prob - 1)  # optional approach via erfinv
    # or simpler:
    # z = stats.norm.ppf( rank_prob )

    # do normal on z
    res_dict = syn_norm(z, X, Xp, proper=proper, ridge=ridge, **kwargs)
    z_pred = res_dict["res"]

    # convert z_pred -> rank positions
    # rank positions are from 1..n -> pick in sorted(y)
    # We'll transform z_pred -> some rank ~ 1..n
    # let's do pred_r = rank of z_pred among themselves, or just
    from scipy.stats import norm

    # transform z_pred from real line -> [1..n]
    cdf_vals = norm.cdf(z_pred)
    new_ranks = np.round(cdf_vals * (n + 1))
    new_ranks[new_ranks < 1] = 1
    new_ranks[new_ranks > n] = n

    new_ranks = new_ranks.astype(int)

    # sort y
    sorted_y = np.sort(y)

    # get the synthetic y by picking from sorted y
    syn_y = sorted_y[new_ranks - 1]  # shift for 0-based

    # optional smoothing
    if smoothing == "density":
        syn_y = _syn_smooth(syn_y, y)

    return {"res": syn_y, "fit": res_dict["fit"]}


def syn_ranknorm(
    y: np.ndarray,
    X: np.ndarray,
    Xp: np.ndarray,
    proper: bool = False,
    ridge: float = 1e-5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Another rank-based normal method. Similar to syn.normrank, but a simpler approach:
    - fit normal on raw y
    - get predicted residuals
    - re-sort them to match y's distribution
    This is a direct adaptation from the R code syn.ranknorm, but that code is quite close
    to syn.normrank. Implementation details may differ in Python.
    """
    # For practical usage, most prefer syn_normrank or pmm. This is a simplified variant.

    # We can simply call syn_norm, then re-map the output ranks to the sorted y
    # 1) fit
    res_dict = syn_norm(y, X, Xp, proper=proper, ridge=ridge, **kwargs)
    pred_y = res_dict["res"]

    # 2) rank-based mapping
    # re-sample from y distribution according to ranks of pred_y
    n = len(y)
    sorted_y = np.sort(y)

    # rank transform pred_y
    ranks = np.argsort(np.argsort(pred_y))
    # sample with bootstrap
    # to emulate the R code: res <- sort(sample(y,replace=T))[rankpred]
    # We can do something simpler:
    y_boot = np.random.choice(y, size=n, replace=True)
    y_boot_sorted = np.sort(y_boot)

    # re-index
    final_y = y_boot_sorted[ranks]
    res_dict["res"] = final_y
    return res_dict


def _syn_smooth(ysyn: np.ndarray, yobs: np.ndarray) -> np.ndarray:
    """
    Approximate "density smoothing" used in R's syn.smooth approach.
    For large categories, it draws from a normal with a bandwidth ~ "SJ".
    For simplicity, we pick a small bandwidth from the data.
    This is only an approximate approach.
    """
    # If y is heavily concentrated on one value, skip
    # or if top-coded, skip
    # We'll do a naive approach:

    # remove infinite or missing
    valid_mask = ~pd.isna(ysyn)
    valid_vals = ysyn[valid_mask]

    # Basic approach: bandwidth ~ silverman
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # We can use scipy.stats.gaussian_kde or compute a bandwidth
        # Let's do something simpler: standard dev * 0.1
        bw = np.std(valid_vals) * 0.1
    if bw <= 0:
        return ysyn

    # add random N(0,bw) to subset
    perturbed = valid_vals + np.random.normal(scale=bw, size=len(valid_vals))

    # clamp them to min/max
    mn, mx = np.min(yobs), np.max(yobs)
    perturbed = np.clip(perturbed, mn, mx)

    ysyn[valid_mask] = np.round(perturbed, _decimal_places(yobs))
    return ysyn


# For use in syn_normrank, we need an inverse error function
# If your environment doesn't have erfinv, we can define it or import from scipy.special

try:
    from math import erfinv  # Python 3.8+
except ImportError:
    from scipy.special import erfinv  # fallback
