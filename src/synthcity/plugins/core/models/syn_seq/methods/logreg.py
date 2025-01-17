# logreg.py
"""
syn_logreg: Fit a logistic or linear regression model for `y` given `X`, 
            then generate new `y` values from the fitted model.
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression

def syn_logreg(y, X, random_state=0, is_classification=True, **kwargs):
    """
    Fit a logistic regression or linear regression (depending on the `is_classification` flag).
    
    Args:
        y: 1D array-like, shape (n_samples,). The target values.
        X: 2D array-like, shape (n_samples, n_features). Predictors.
        random_state: int. Random seed for reproducibility.
        is_classification: bool. If True, use LogisticRegression. If False, use LinearRegression.
        **kwargs: Additional keyword arguments passed to the underlying scikit-learn model.
        
    Returns:
        model: dict with fields:
            - "name": "logreg"
            - "model_type": "classification" or "regression"
            - "regressor": The actual fitted sklearn model
            - "unique_labels": For classification only, an array of unique labels
    """
    # Ensure arrays
    y_arr = np.asarray(y)
    X_arr = np.asarray(X)

    # Simple example: drop any row where y is NaN
    valid_mask = ~pd.isna(y_arr)
    y_valid = y_arr[valid_mask]
    X_valid = X_arr[valid_mask]

    if is_classification:
        # Fit logistic regression
        clf = LogisticRegression(random_state=random_state, **kwargs)
        clf.fit(X_valid, y_valid)

        # We keep track of the labels in case we need them for sampling
        unique_labels = np.unique(y_valid)
        fitted_model = {
            "name": "logreg",
            "model_type": "classification",
            "regressor": clf,
            "unique_labels": unique_labels
        }
    else:
        # Fit linear regression
        reg = LinearRegression(**kwargs)
        reg.fit(X_valid, y_valid)

        fitted_model = {
            "name": "logreg",
            "model_type": "regression",
            "regressor": reg,
            # For regression, we typically don't store "unique_labels"
        }

    return fitted_model


def generate_logreg(fitted_logreg, X_new, random_state=0, **kwargs):
    """
    Generate synthetic y-values given new X using the fitted model.
    
    Args:
        fitted_logreg: dict returned by syn_logreg. Must contain:
            - "model_type" (classification or regression)
            - "regressor"  (the sklearn fitted model)
            - "unique_labels" if classification
        X_new: 2D array-like of shape (m_samples, n_features).
        random_state: int. Random seed if sampling is involved.
        **kwargs: Additional arguments, e.g.:
            - 'sample_prob' (bool) if you want to sample from predicted probabilities 
              for classification.
    
    Returns:
        y_syn: 1D array-like, shape (m_samples,). The generated target values.
    """
    rng = np.random.default_rng(random_state)

    model_type = fitted_logreg.get("model_type", "classification")
    model = fitted_logreg["regressor"]

    X_arr = np.asarray(X_new)
    n = X_arr.shape[0]

    if model_type == "classification":
        # By default, we sample from predicted probabilities
        # unless user wants just the predicted class
        sample_from_prob = kwargs.get("sample_prob", True)

        if sample_from_prob:
            # Probabilistic sample from predicted probabilities
            probs = model.predict_proba(X_arr)
            labels = fitted_logreg["unique_labels"]

            y_syn = np.empty(n, dtype=labels.dtype)
            for i in range(n):
                y_syn[i] = rng.choice(labels, p=probs[i])
        else:
            # Deterministic approach: just predict the label
            y_syn = model.predict(X_arr)

    else:
        # Regression => normal numeric predictions
        # (Potentially you could add noise or round if you want discrete ints)
        y_syn = model.predict(X_arr)

    return np.array(y_syn)
