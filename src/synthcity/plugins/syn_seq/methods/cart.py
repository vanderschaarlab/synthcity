# synthcity/plugins/syn_seq/methods/cart.py

"""
Implements CART-based data synthesis, analogous to R's syn.cart from the synthpop package.

Key ideas from R syn.cart:
    - If y is numeric, fit a regression tree (method="anova"), and for each new Xp row,
      find the corresponding leaf node, then randomly sample from the original y values in that leaf.
    - If y is categorical, fit a classification tree (method="class"), and for each new Xp row,
      generate a synthetic value by sampling from the original y distribution within that leaf.
    - Optional smoothing for numeric y (like syn.smooth approach) can be applied if needed.
      This adds some random noise around the predicted values.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# scikit-learn for CART
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

try:
    from .misc import syn_smooth
except ImportError:
    # fallback if we have not created misc.py or the function in the current environment
    def syn_smooth(ysyn: np.ndarray, yobs: np.ndarray) -> np.ndarray:
        """
        Simple fallback for smoothing if 'syn_smooth' is not yet implemented in misc.py
        """
        return ysyn  # no-op fallback


def syn_cart(
    y: np.ndarray,
    X: np.ndarray,
    Xp: np.ndarray,
    smoothing: str = "",
    proper: bool = False,
    # parallels R's "minbucket" and "cp" arguments
    min_samples_leaf: int = 5,
    max_leaf_nodes: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Synthesis by fitting a CART (decision tree) and sampling from the leaf nodes.

    Args:
        y: (n,) array of response values (numeric or categorical).
        X: (n, p) array for the training data's covariates.
        Xp: (m, p) array for the new data (the rows we want to synthesize).
        smoothing: str, optional. If nonempty and y is numeric, we can optionally
            add a "syn_smooth" style approach for the final sampled values.
            Example: "density" => calls syn_smooth logic on numeric predictions.
        proper: bool. If True, can bootstrap the original data prior to fitting
            or do other "proper" approaches. Currently, we do not do
            a separate bootstrap by default. The R code uses a condition for that,
            but let's keep it simpler.
        min_samples_leaf: int. Minimum samples in each leaf (similar to R's minbucket).
        max_leaf_nodes: Optional[int]. If set, limits the maximum number of leaf nodes.
        random_state: random seed for reproducibility.
        **kwargs: Additional arguments for the scikit-learn DecisionTree models.

    Returns:
        A dictionary:
          {
            "res": array of shape (m,), the synthetic y for each row in Xp
            "fit": the fitted tree or relevant information
          }
    """

    # 1) Convert to np arrays
    y = np.array(y)
    X = np.array(X)
    Xp = np.array(Xp)

    # 2) Check if y is numeric or categorical
    is_numeric = True
    if pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y):
        is_numeric = False
    elif pd.api.types.is_integer_dtype(y) or pd.api.types.is_float_dtype(y):
        # still numeric
        pass
    else:
        # fallback if not recognized
        # check if y has many repeated categories => treat as factor
        # But typically you can do: is_numeric = np.issubdtype(y.dtype, np.number)
        # We'll just do a final fallback:
        is_numeric = np.issubdtype(y.dtype, np.number)

    # (Optional) if proper, we might want to bootstrap. We skip that for now:
    if proper:
        # simple sample indices with replacement
        n = len(y)
        indices = np.random.RandomState(random_state).choice(n, size=n, replace=True)
        y = y[indices]
        X = X[indices, :]

    # 3) Fit the CART
    if is_numeric:
        # regression tree
        cart_model = DecisionTreeRegressor(
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            **kwargs
        )
    else:
        # classification tree
        cart_model = DecisionTreeClassifier(
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            **kwargs
        )

    cart_model.fit(X, y)

    # 4) For each row in Xp, find the leaf node
    # scikit-learn does not directly return node IDs in `predict`, but we have "apply()"
    # which returns the leaf node ID for each sample.
    leaf_ids_syn = cart_model.apply(Xp)

    # 5) For each leaf ID, gather the original y from that leaf, then sample from it
    # For that, we also need the leaf ID for the training data
    leaf_ids_obs = cart_model.apply(X)

    # Precompute a dict: leaf_id -> array of y
    leaf_dict = {}
    for leaf_id in np.unique(leaf_ids_obs):
        mask = (leaf_ids_obs == leaf_id)
        leaf_dict[leaf_id] = y[mask]

    syn_y = []
    rng = np.random.RandomState(random_state)

    for leaf_id in leaf_ids_syn:
        donors = leaf_dict[leaf_id]
        if len(donors) == 0:
            # fallback: if a leaf is empty for some reason, pick from entire y
            newval = rng.choice(y)
        else:
            newval = rng.choice(donors)
        syn_y.append(newval)

    syn_y = np.array(syn_y)

    # 6) If numeric and smoothing="density", apply some smoothing
    if is_numeric and smoothing == "density":
        syn_y = syn_smooth(syn_y, y)

    return {"res": syn_y, "fit": cart_model}
