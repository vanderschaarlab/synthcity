# ctree.py
# stdlib
from typing import Any, Dict, List, Union

# third party
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def syn_ctree(
    y: Any,
    X: Any,
    random_state: int = 0,
    is_classification: bool = False,
    max_depth: Union[int, None] = None,
    min_samples_leaf: int = 1,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Fit a conditional inference tree model for `y` given `X`, in a style akin to
    'conditional inference trees' via scikit-learn's DecisionTreeClassifier/Regressor.

    Args:
        y: 1D array-like for the target variable (numeric or categorical).
        X: 2D array-like for predictor features of shape (n_samples, n_features).
        random_state: integer seed for reproducibility.
        is_classification: if True, train a classifier. Otherwise, regress.
        max_depth: optional int, maximum depth of the tree.
        min_samples_leaf: optional int, minimum samples per leaf.
        **kwargs: additional parameters passed to the scikit-learn decision tree.

    Returns:
        A dict containing:
            "name": "ctree"
            "model": the trained DecisionTree model.
            "is_classification": bool indicating classification vs. regression.
            "leaf_index_map": array-like of sample indices for each leaf node.
            "train_X": the original training X (for direct leaf-based sampling).
            "train_y": the original training y.
    """
    # Convert X, y to numpy arrays if needed
    X = np.asarray(X)
    y = np.asarray(y)

    # Remove rows with NaN in y (simple approach; adapt if needed)
    valid_mask = ~pd.isna(y)
    y = y[valid_mask]
    X = X[valid_mask, :]

    # Choose classifier or regressor
    if is_classification:
        tree = DecisionTreeClassifier(
            random_state=random_state,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            **kwargs
        )
    else:
        tree = DecisionTreeRegressor(
            random_state=random_state,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            **kwargs
        )

    # Fit
    tree.fit(X, y)

    # For each training sample, find which leaf it ends in
    leaf_ids = tree.apply(X)  # shape (n_samples,)
    unique_leaf_ids = np.unique(leaf_ids)

    # Build an index map: for each leaf id, which training indices are in that leaf?
    leaf_index_map: List[np.ndarray] = []
    for leaf_id in unique_leaf_ids:
        idxs = np.where(leaf_ids == leaf_id)[0]
        leaf_index_map.append(idxs)

    model: Dict[str, Any] = {
        "name": "ctree",
        "model": tree,
        "is_classification": is_classification,
        "leaf_index_map": leaf_index_map,
        "train_X": X,
        "train_y": y,
    }
    return model


def generate_ctree(
    fitted_ctree: Dict[str, Any], X_new: Any, random_state: int = 0, **kwargs: Any
) -> np.ndarray:
    """
    Generate y values from the fitted ctree model by sampling from leaf-level distributions.

    Args:
        fitted_ctree: a dict returned by syn_ctree(...).
        X_new: 2D array-like (n_samples, n_features) to generate new y's for.
        random_state: integer seed for reproducibility.
        **kwargs: unused here, but kept for interface consistency.

    Returns:
        A 1D numpy array of generated y values. Classification => random draws
        from the leaf's classes, Regression => random draws from the leaf's numeric distribution.
    """
    rng = np.random.default_rng(random_state)

    tree = fitted_ctree["model"]
    leaf_index_map = fitted_ctree["leaf_index_map"]
    is_classification = fitted_ctree["is_classification"]
    train_X = fitted_ctree["train_X"]
    train_y = fitted_ctree["train_y"]

    # Which leaf does each X_new sample fall into?
    X_new = np.asarray(X_new)
    new_leaf_ids = tree.apply(X_new)

    # Build a dict: leaf_id -> position in leaf_index_map
    train_leaf_ids = tree.apply(train_X)
    unique_leaf_ids = np.unique(train_leaf_ids)
    leafid_to_pos = {leaf_id: i for i, leaf_id in enumerate(unique_leaf_ids)}

    y_syn: List[Any] = []
    for leaf_id in new_leaf_ids:
        if leaf_id not in leafid_to_pos:
            # fallback if leaf_id wasn't seen during training
            # e.g. out-of-distribution input => just do a normal predict
            y_syn.append(tree.predict(X_new[[0]])[0])
            continue

        pos = leafid_to_pos[leaf_id]
        idxs_in_leaf = leaf_index_map[pos]

        if len(idxs_in_leaf) == 0:
            # fallback
            y_syn.append(tree.predict(X_new[[0]])[0])
            continue

        # pull the actual training labels from this leaf
        leaf_labels = train_y[idxs_in_leaf]

        if is_classification:
            # discrete random draw from leaf_labels
            chosen_idx = rng.integers(0, len(leaf_labels))
            y_syn.append(leaf_labels[chosen_idx])
        else:
            # numeric => random draw from leaf_labels
            chosen_idx = rng.integers(0, len(leaf_labels))
            y_syn.append(leaf_labels[chosen_idx])

    return np.array(y_syn)
