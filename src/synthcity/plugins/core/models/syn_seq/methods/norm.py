# stdlib
from typing import Any, Dict

# third party
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def syn_norm(
    y: np.ndarray, X: pd.DataFrame, random_state: int = 0, **kwargs: Any
) -> Dict[str, Any]:
    """
    Fit a simple linear model (y ~ X) and estimate residual standard deviation.
    The fitted model is stored in a dictionary that can be used later for generation.

    Parameters
    ----------
    y : np.ndarray
        Target values, must be numeric.
    X : pd.DataFrame
        Predictor matrix.
    random_state : int, default=0
        Seed for random number generator.
    **kwargs : dict
        Additional parameters. For example:
          - add_noise : bool, default=True
              Whether to add residual Gaussian noise at generation time.
          - noise_scale : float, default=1.0
              Scale factor for the residual standard deviation when generating noise.

    Returns
    -------
    model : dict
        A dictionary storing fitted parameters:
          {
            "name": "norm",
            "coef_": np.array of shape (n_features,),
            "intercept_": float,
            "resid_std_": float,
            "random_state": int,
            "add_noise": bool,
            "noise_scale": float
          }
    """
    add_noise = kwargs.get("add_noise", True)
    noise_scale = kwargs.get("noise_scale", 1.0)

    # Fit linear regression
    reg = LinearRegression()
    reg.fit(X, y)

    # Estimate residual standard deviation
    y_pred = reg.predict(X)
    valid_mask = ~pd.isna(y_pred) & ~pd.isna(y)
    residuals = y[valid_mask] - y_pred[valid_mask]
    if len(residuals) > 1:
        resid_std_ = np.sqrt(np.sum(residuals**2) / (len(residuals) - 1))
    else:
        resid_std_ = 0.0

    model: Dict[str, Any] = {
        "name": "norm",
        "coef_": reg.coef_,
        "intercept_": reg.intercept_,
        "resid_std_": resid_std_,
        "random_state": random_state,
        "add_noise": add_noise,
        "noise_scale": noise_scale,
    }
    return model


def generate_norm(
    fitted_norm: Dict[str, Any], X_new: pd.DataFrame, **kwargs: Any
) -> np.ndarray:
    """
    Generate synthetic predictions using the fitted 'norm' model.

    Parameters
    ----------
    fitted_norm : dict
        The fitted model dictionary returned by syn_norm().
    X_new : pd.DataFrame
        Predictor matrix for which to generate synthetic y values.
    **kwargs : dict
        Additional parameters. If provided, can override:
          - add_noise : bool
          - noise_scale : float

    Returns
    -------
    y_syn : np.ndarray
        Synthetic numeric predictions of shape (n_samples,).
    """
    # Extract the stored parameters
    coef_ = fitted_norm["coef_"]
    intercept_ = fitted_norm["intercept_"]
    resid_std_ = fitted_norm["resid_std_"]
    random_state = fitted_norm["random_state"]

    # Default to stored settings; allow overrides
    add_noise = kwargs.get("add_noise", fitted_norm["add_noise"])
    noise_scale = kwargs.get("noise_scale", fitted_norm["noise_scale"])

    # Predict the mean
    y_mean = np.dot(X_new, coef_) + intercept_

    # Optionally add noise
    if add_noise and resid_std_ > 0:
        rng = np.random.RandomState(random_state)
        noise = rng.normal(loc=0.0, scale=resid_std_ * noise_scale, size=len(y_mean))
        y_syn = y_mean + noise
    else:
        y_syn = y_mean

    return np.array(y_syn)
