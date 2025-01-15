# synthcity/plugins/syn_seq/methods/cart.py

"""
Implementation of CART-based data synthesis.

This approach fits a decision tree (regression or classification) on the observed data.
Then, for each new input row, we locate the corresponding leaf node and sample the
synthetic values from the observed data points that ended up in that leaf node.

If the target is numeric, a regression tree is used.
If the target is categorical, a classification tree is used.

Optionally, you can apply a simple smoothing step for numeric targets by specifying
`smoothing="density"`.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

try:
    from .misc import syn_smooth
except ImportError:
    # Fallback if not found
    def syn_smooth(ysyn: np.ndarray, yobs: np.ndarray) -> np.ndarray:
        return ysyn  # no-op fallback


def syn_cart(
    y: np.ndarray,
    X: np.ndarray,
    Xp: np.ndarray,
    smoothing: str = "",
    proper: bool = False,
    min_samples_leaf: int = 5,
    max_leaf_nodes: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    CART-based data synthesis.

    Parameters
    ----------
    y : np.ndarray
        The target array of shape (n,). Can be numeric or categorical.
    X : np.ndarray
        Feature array of shape (n, p) for the training data.
    Xp : np.ndarray
        Feature array of shape (m, p) for which to generate synthetic targets.
    smoothing : str, optional
        If "density" and y is numeric, apply a small smoothing to the final sampled values.
    proper : bool, optional
        If True, apply a bootstrap resample on (X, y) before fitting the tree.
    min_samples_leaf : int, optional
        Minimum number of samples per leaf.
    max_leaf_nodes : int, optional
        If not None, maximum number of leaf nodes.
    random_state : int, optional
        Random seed for reproducibility.
    **kwargs : Any
        Additional keyword arguments passed to the DecisionTreeRegressor/Classifier.

    Returns
    -------
    result : dict
        A dictionary containing:
          - "res": a (m,) array of synthetic targets for Xp
          - "fit": the fitted decision tree model
    """
    # Convert to numpy arrays if not already
    y = np.asarray(y)
    X = np.asarray(X)
    Xp = np.asarray(Xp)

    # Determine if y is numeric or categorical
    if pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y):
        is_numeric = False
    elif pd.api.types.is_integer_dtype(y) or pd.api.types.is_float_dtype(y):
        is_numeric = True
    else:
        is_numeric = np.issubdtype(y.dtype, np.number)

    # Optional "proper" synthesis: bootstrap (X, y)
    if proper:
        n = len(y)
        rng_boot = np.random.RandomState(random_state)
        indices = rng_boot.choice(n, size=n, replace=True)
        y = y[indices]
        X = X[indices, :]

    # Fit a CART model (regression or classification)
    if is_numeric:
        cart_model = DecisionTreeRegressor(
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            **kwargs
        )
    else:
        cart_model = DecisionTreeClassifier(
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            **kwargs
        )

    cart_model.fit(X, y)

    # Leaf assignments for new data
    leaf_ids_syn = cart_model.apply(Xp)
    # Leaf assignments for observed data
    leaf_ids_obs = cart_model.apply(X)

    # Build a dict: leaf_id -> original y values in that leaf
    leaf_dict = {}
    for leaf_id in np.unique(leaf_ids_obs):
        mask = (leaf_ids_obs == leaf_id)
        leaf_dict[leaf_id] = y[mask]

    # Generate synthetic y values by sampling from the leaf distributions
    rng = np.random.RandomState(random_state)
    syn_y = []
    for leaf_id in leaf_ids_syn:
        donors = leaf_dict.get(leaf_id, [])
        if len(donors) == 0:
            # Fallback to full y if no donors found
            new_val = rng.choice(y)
        else:
            new_val = rng.choice(donors)
        syn_y.append(new_val)
    syn_y = np.array(syn_y)

    # Optional smoothing for numeric
    if is_numeric and smoothing == "density":
        syn_y = syn_smooth(syn_y, y)

    return {"res": syn_y, "fit": cart_model}
