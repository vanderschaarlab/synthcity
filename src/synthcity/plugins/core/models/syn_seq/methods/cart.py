# cart.py
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


def syn_cart(y, X, random_state=0, **kwargs):
    """
    Fit a CART model for the target `y` given predictors `X`.
    Return a dictionary containing the fitted regressor/classifier or relevant info.

    This implementation also stores each leaf node's distribution of y-values,
    so that generate_cart() can sample from them rather than just do a point prediction.
    """

    # Convert y, X to numpy if needed
    X = np.asarray(X)
    y = np.asarray(y)

    # Heuristic: decide classification vs. regression
    y_series = pd.Series(y)
    unique_vals = y_series.dropna().unique()
    is_classification = False
    if y_series.dtype.name in ["object", "category"]:
        is_classification = True
    elif len(unique_vals) < 15 and y_series.dtype.kind in ["i", "u"]:
        # small unique integer => classification
        is_classification = True

    # Build the tree
    if is_classification:
        estimator = DecisionTreeClassifier(
            random_state=random_state, **kwargs
        )
    else:
        estimator = DecisionTreeRegressor(
            random_state=random_state, **kwargs
        )
    estimator.fit(X, y)

    # Map each training row to its leaf index
    leaf_ids = estimator.apply(X)
    # Collect y-values in each leaf
    leaf_indexed_y = {}
    for lid, val in zip(leaf_ids, y):
        if lid not in leaf_indexed_y:
            leaf_indexed_y[lid] = []
        leaf_indexed_y[lid].append(val)
    # Convert to arrays
    for lid in leaf_indexed_y:
        leaf_indexed_y[lid] = np.array(leaf_indexed_y[lid])

    model = {
        "name": "cart",
        "estimator": estimator,
        "leaf_indexed_y": leaf_indexed_y,
        "is_classification": is_classification,
    }
    return model


def generate_cart(fitted_cart, X_new, random_state=0, **kwargs):
    """
    Use the fitted cart model to generate predicted y's or do custom sequential sampling.
    Return an array-like of predictions.

    Here, for each row in X_new, we:
      - identify the leaf node via .apply(...)
      - randomly sample from that leaf's empirical distribution (leaf_indexed_y).

    If the leaf is unknown (e.g. corner case), we fallback to sampling from the entire training distribution.
    """
    estimator = fitted_cart["estimator"]
    leaf_indexed_y = fitted_cart["leaf_indexed_y"]

    # Convert X_new to numpy if needed
    X_new = np.asarray(X_new)

    # Identify leaves
    leaf_ids = estimator.apply(X_new)

    # For reproducibility
    rng = np.random.default_rng(random_state)

    y_syn = []
    # Precompute a global fallback if needed
    all_vals = np.concatenate(list(leaf_indexed_y.values())) if leaf_indexed_y else []

    for lid in leaf_ids:
        if lid in leaf_indexed_y:
            vals_in_leaf = leaf_indexed_y[lid]
            idx = rng.integers(len(vals_in_leaf))
            y_syn.append(vals_in_leaf[idx])
        else:
            # Fallback case
            if len(all_vals) == 0:
                y_syn.append(np.nan)
            else:
                idx = rng.integers(len(all_vals))
                y_syn.append(all_vals[idx])

    return np.array(y_syn)
