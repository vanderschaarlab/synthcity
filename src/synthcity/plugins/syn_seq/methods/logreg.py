# synthcity/plugins/syn_seq/methods/logreg.py

"""
Implements a logistic regression synthesis approach, analogous (in spirit)
to R's `syn.logreg` from synthpop, using scikit-learn's LogisticRegression.

Key ideas:
    - If denom is None => treat y as binary {0,1} or {False,True} and fit standard logistic regression.
      Then for each row in Xp, we predict probability p and draw from Bernoulli(p).
    - If denom is not None => treat (y, denom) as a binomial outcome (y successes out of denom trials).
      We'll mimic this by fitting logistic regression on y/denom with sample_weight=denom in scikit-learn.
      Then for Xp, we predict probability p and draw from Binomial(denomp, p).
      (Here denomp is analogous to how many trials for each new observation.)
    - If proper=True => we bootstrap (X, y) before fitting, as in the R code.

Example usage:
    result = syn_logreg(
        y, X, Xp,
        denom=None,
        denomp=None,
        proper=False,
        random_state=42,
        ...
    )
    y_syn = result["res"]
    model_info = result["fit"]
    ...
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def syn_logreg(
    y: Union[pd.Series, np.ndarray],
    X: Union[pd.DataFrame, np.ndarray],
    Xp: Union[pd.DataFrame, np.ndarray],
    denom: Optional[Union[pd.Series, np.ndarray]] = None,
    denomp: Optional[Union[pd.Series, np.ndarray]] = None,
    proper: bool = False,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Synthetic data generation using logistic regression.

    Args:
        y: 1D array-like of shape (n,). The response variable.
           If denom=None => y should be {0,1} for standard logistic.
           If denom is provided => y is # successes, and denom is # trials.
        X: 2D array-like of shape (n, p). Covariates for training data.
        Xp: 2D array-like of shape (m, p). Covariates for new data to generate synthetic y.
        denom: optional array-like. If provided => treat as binomial with y successes out of denom.
        denomp: optional array-like. If provided => used for the new data's # trials, default is None => 1 trial.
        proper: bool, default=False. If True, bootstrap (X,y) prior to training.
        random_state: optional int for reproducibility.
        **kwargs: Additional arguments for LogisticRegression, e.g. `max_iter=1000`.

    Returns:
        A dictionary with:
            "res": array-like of shape (m,) with the synthetic outcome for each row in Xp.
            "fit": dictionary containing the fitted logistic model and metadata.

    Notes:
        - We rely on scikit-learn's logistic regression, which doesn't natively handle binomial
          responses with multiple trials. We approximate by using y/denom as the label
          and sample_weight=denom.
        - If denom is not None, we interpret the new data's # trials as `denomp`. Then we draw from
          Binomial(denomp, p).
        - If denom=None => standard Bernoulli( p ) draw.
        - "proper" synthesis can be done by bootstrap resampling (X, y) before fitting.
    """
    rng = np.random.RandomState(random_state)

    # Convert input to arrays
    if isinstance(y, pd.Series):
        y = y.values
    X = np.asarray(X)
    Xp = np.asarray(Xp)
    n = len(y)

    # If proper => bootstrap
    if proper:
        idx_boot = rng.choice(n, size=n, replace=True)
        X = X[idx_boot, :]
        y = y[idx_boot]
        if denom is not None:
            denom = np.asarray(denom)
            denom = denom[idx_boot]

    # Decide how to fit the logistic model
    if denom is None:
        # Standard logistic with y in {0,1}
        # Some checks or conversions if needed
        model = LogisticRegression(random_state=rng, **kwargs)
        model.fit(X, y)
        # Predict probabilities for new data
        p_pred = model.predict_proba(Xp)[:, 1]
        # Synthetic draws
        y_syn = rng.binomial(1, p_pred)
    else:
        # Binomial with sample_weight = denom
        denom = np.asarray(denom)
        # Convert y/denom => "proportions"
        y_frac = y / denom
        model = LogisticRegression(random_state=rng, **kwargs)
        model.fit(X, y_frac, sample_weight=denom)
        # For new data, we need # trials => denomp
        if denomp is None:
            # If user doesn't specify, default = 1 trial => Bernoulli
            denomp = np.ones(len(Xp), dtype=int)
        else:
            denomp = np.asarray(denomp)
            if len(denomp) != len(Xp):
                raise ValueError("Length of denomp must match length of Xp")

        p_pred = model.predict_proba(Xp)[:, 1]
        # Synthetic draws from Binomial(denomp, p_pred)
        y_syn = np.array([rng.binomial(denomp[i], p_pred[i]) for i in range(len(Xp))])

    return {
        "res": y_syn,
        "fit": {
            "model": model,
            "is_binomial": (denom is not None),
        },
    }
