# stdlib
import warnings
from typing import Any, Dict

# third party
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def syn_rf(
    y: np.ndarray, X: np.ndarray, random_state: int = 0, **kwargs: Any
) -> Dict[str, Any]:
    """
    Fit a RandomForest model in either regression or classification mode,
    depending on whether 'y' is numeric or categorical.

    Args:
        y: 1D array of target values
        X: 2D array of shape (n_samples, n_features)
        random_state: random seed
        kwargs: additional hyperparameters for the RandomForest

    Returns:
        A dictionary containing:
            {
              "name": "rf",
              "is_classifier": bool,
              "rf": fitted RF object,
              "classes_": array of class labels (if classifier),
              ...
            }
    """
    # Default hyperparameters, can be overridden by kwargs
    n_estimators = kwargs.get("n_estimators", 100)
    max_depth = kwargs.get("max_depth", None)
    min_samples_leaf = kwargs.get("min_samples_leaf", 1)
    # etc. add or remove as needed

    # Basic detection logic for classification vs. regression.
    # You may refine this (e.g. check number of unique values, etc.)
    if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_integer_dtype(y):
        # If floating numeric => treat as regression
        is_classifier = False
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    else:
        # otherwise treat as classification
        is_classifier = True
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    # Fit the model
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        rf_model.fit(X, y)

    # Prepare the output dictionary
    model: Dict[str, Any] = {
        "name": "rf",
        "is_classifier": is_classifier,
        "rf": rf_model,
    }

    # If classifier, store class labels for sampling
    if is_classifier:
        model["classes_"] = rf_model.classes_

    return model


def generate_rf(
    fitted_rf: Dict[str, Any], X_new: np.ndarray, **kwargs: Any
) -> np.ndarray:
    """
    Generate synthetic target values using a previously fitted RF model.

    Args:
        fitted_rf: dict returned by syn_rf(...).
        X_new: 2D array of shape (n_samples, n_features) for which we want predictions
        kwargs: any additional settings (e.g., sampling strategies)

    Returns:
        y_syn: synthetic target array
    """
    rf_model = fitted_rf["rf"]
    is_classifier = fitted_rf["is_classifier"]

    if is_classifier:
        # For classification, sample from predicted probability distribution
        probas = rf_model.predict_proba(X_new)
        classes = fitted_rf["classes_"]
        sampled_indices = [
            np.random.choice(len(classes), p=probas[i]) for i in range(len(X_new))
        ]
        y_syn = np.array([classes[idx] for idx in sampled_indices])
    else:
        # For regression, just return the raw numeric predictions
        y_syn = rf_model.predict(X_new)

    return np.array(y_syn)
