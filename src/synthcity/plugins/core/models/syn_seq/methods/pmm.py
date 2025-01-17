# File: pmm.py

import numpy as np
from sklearn.linear_model import LinearRegression

def syn_pmm(y, X, random_state=0, k=5, use_intercept=True, **kwargs):
    """
    Fit a PMM (Predictive Mean Matching) model.

    Args:
        y: shape (n_samples,) - target array
        X: shape (n_samples, n_features) - predictor variables
        random_state: random seed
        k: number of nearest neighbors to match
        use_intercept: whether or not to fit an intercept in the regression
        kwargs: additional arguments to pass into the regressor if needed

    Returns:
        A dictionary representing the fitted PMM model, containing:
            - "name" : str
            - "model": the regression model used to predict
            - "X": training features
            - "y": training targets
            - "y_hat": predicted values on training set
            - "k": number of neighbors
            - "random_state": the random seed used
    """
    rng = np.random.default_rng(random_state)

    # For demonstration, use a linear regressor. Modify or replace as needed.
    regressor = LinearRegression(fit_intercept=use_intercept, **kwargs)
    regressor.fit(X, y)
    y_hat = regressor.predict(X)

    model = {
        "name": "pmm",
        "model": regressor,
        "X": X,
        "y": y,
        "y_hat": y_hat,
        "k": k,
        "random_state": random_state
    }
    return model


def generate_pmm(fitted_pmm, X_new, random_state=None, **kwargs):
    """
    Generate new synthetic values using the fitted PMM model.

    For each row in X_new:
       1) Predict the mean outcome (via the regression model).
       2) Find the k closest samples in the training set
          (based on distance in predicted space).
       3) Randomly pick one neighbor among those k to get the actual y.

    Args:
        fitted_pmm: dictionary from syn_pmm(...)
        X_new: shape (m, n_features) for which to generate new y
        random_state: random seed override (optional)
        kwargs: additional arguments (not used here, but available for extension)

    Returns:
        y_syn: shape (m,) - synthetic target values
    """
    if random_state is None:
        random_state = fitted_pmm.get("random_state", 0)
    rng = np.random.default_rng(random_state)

    regressor = fitted_pmm["model"]
    X_train = fitted_pmm["X"]
    y_train = fitted_pmm["y"]
    y_hat_train = fitted_pmm["y_hat"]
    k = fitted_pmm["k"]

    # Predict for the new data
    y_hat_new = regressor.predict(X_new)

    # Container for synthetic outcomes
    y_syn = np.empty(len(y_hat_new), dtype=y_train.dtype)

    # PMM logic
    for i, pred_val in enumerate(y_hat_new):
        # Distance in predicted space
        dist = np.abs(y_hat_train - pred_val)
        # Indices of the k smallest distances
        neighbor_idxs = np.argpartition(dist, kth=k)[:k]
        # Pick one random neighbor among these k
        chosen_idx = rng.choice(neighbor_idxs)
        # Use that neighbor's actual y
        y_syn[i] = y_train[chosen_idx]

    return np.array(y_syn)
