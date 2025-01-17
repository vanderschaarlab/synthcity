# File: polyreg.py

import numpy as np
from sklearn.linear_model import LogisticRegression

def syn_polyreg(y, X, random_state=0, solver="lbfgs", max_iter=200, **kwargs):
    """
    Fit polynomial (polytomous) regression or polynomial expansions.

    In the context of the R package 'synthpop', "polyreg" typically
    refers to polytomous (multinomial) logistic regression for
    categorical outcomes. This function sets up a multinomial logistic
    regression model in scikit-learn to mimic that functionality.

    Parameters
    ----------
    y : np.ndarray
        The target array (categorical or numeric-encoded).
    X : np.ndarray
        The predictor array of shape (n_samples, n_features).
    random_state : int, optional
        Random seed for reproducibility.
    solver : str, optional
        Solver to use in the LogisticRegression. Typically 'lbfgs' is
        suitable for multi_class='multinomial'.
    max_iter : int, optional
        Maximum number of iterations for solver convergence.
    **kwargs
        Additional keyword arguments passed to the LogisticRegression.

    Returns
    -------
    fitted_polyreg : dict
        A dictionary containing:
        - "name": fixed string "polyreg".
        - "model": the fitted LogisticRegression object.
        - "random_state": the random seed used.
        - any other configuration parameters you want to store.
    """
    model = LogisticRegression(
        multi_class="multinomial",
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        **kwargs
    )
    model.fit(X, y)

    # Wrap up the fitted model and meta info in a dictionary
    return {
        "name": "polyreg",
        "model": model,
        "random_state": random_state,
        "solver": solver,
        "max_iter": max_iter
    }


def generate_polyreg(fitted_polyreg, X_new, random_state=None, **kwargs):
    """
    Generate synthetic values from the fitted polynomial (polytomous) regression model.

    Parameters
    ----------
    fitted_polyreg : dict
        Dictionary as returned by syn_polyreg().
        Must contain:
          - "name": "polyreg"
          - "model": the trained LogisticRegression object
          - "random_state": int
    X_new : np.ndarray
        The predictor array used to generate synthetic outcomes.
    random_state : int, optional
        If provided, overrides the random seed from fitted_polyreg["random_state"].
    **kwargs
        Additional keyword arguments for flexibility (e.g., temperature scaling).

    Returns
    -------
    y_syn : np.ndarray
        An array of synthetic draws from the fitted distribution.
    """
    model = fitted_polyreg["model"]
    rs = random_state if random_state is not None else fitted_polyreg["random_state"]

    # Predicted probabilities for each row in X_new
    probs = model.predict_proba(X_new)
    classes = model.classes_

    rng = np.random.default_rng(rs)
    n_samples = X_new.shape[0]
    y_syn = []

    for i in range(n_samples):
        draw = rng.choice(classes, p=probs[i])
        y_syn.append(draw)

    return np.array(y_syn)
