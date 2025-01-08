# synthcity/plugins/syn_seq/methods/ctree.py

"""
Implements a conditional inference tree approach, analogous (in spirit) to R's 'ctree' in the party/partykit package.
However, Python does not have a built-in ctree identical to R's version. We approximate with sklearn's DecisionTreeClassifier/Regressor.

Key ideas from R syn.ctree:
    - For a response y (categorical or numeric), fit a conditional inference tree.
    - Each leaf node provides a "donor pool" of observed y's to draw from, or for classification, the leaf can directly yield a predicted category distribution.
    - The new data Xp is assigned to leaf nodes, from which we sample y in a manner consistent with the leaf distribution.

In this Python version:
    - If y is numeric, we use DecisionTreeRegressor, gather all training-set y values in that leaf, and sample one at random (or optionally do smoothing).
    - If y is categorical, we use DecisionTreeClassifier and gather all training-set y values in that leaf, and sample one at random in proportion to the leaf’s distribution.

We call this function syn_ctree. The logic is:
    1. If y is numeric => regressor. If y is factor/categorical => classifier.
    2. If proper=True => bootstrap (X, y) before training.
    3. min_samples_leaf, max_depth, etc. can be specified via kwargs to control tree growth.
    4. After training, for each row in Xp, we find the leaf node. Then we gather the original training y in that same leaf. We sample from them with replacement to produce a synthetic y.

Note: This is a simplified approximation of ctree from R. For more faithful approaches, consider specialized python packages or rpy2-based bridging.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


def syn_ctree(
    y: Union[pd.Series, np.ndarray],
    X: Union[pd.DataFrame, np.ndarray],
    Xp: Union[pd.DataFrame, np.ndarray],
    proper: bool = False,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Synthetic data generation using a 'ctree'-like approach (approximated by sklearn decision trees).

    Args:
        y: 1D array-like of shape (n,). The response variable.
            - If numeric => we treat as regression (DecisionTreeRegressor).
            - If factor/categorical => we treat as classification (DecisionTreeClassifier).
        X: 2D array-like of shape (n, p). Covariates of the training data.
        Xp: 2D array-like of shape (m, p). Covariates of the new data to synthesize for.
        proper: bool, default=False. If True, apply bootstrapping to (X, y) prior to training.
        random_state: Optional[int]. Seed for reproducibility.
        **kwargs: Additional arguments for the sklearn DecisionTree model, such as:
                  - min_samples_leaf=5
                  - max_depth=None
                  - etc.

    Returns:
        A dictionary:
            "res": array of shape (m,) of synthetic outcomes for each row in Xp.
            "fit": a dictionary describing the fitted tree, plus metadata.
    """

    rng = np.random.RandomState(random_state)

    # Convert inputs to arrays
    if isinstance(y, pd.Series):
        # If it's categorical, we detect that below
        y_dtype = y.dtype
        # keep track of categories if categorical
        if pd.api.types.is_categorical_dtype(y):
            y_categories = y.cat.categories
        else:
            y_categories = None
        y = y.values
    else:
        y_categories = None
        y_dtype = None

    X = np.asarray(X)
    Xp = np.asarray(Xp)

    n = len(y)

    # Bootstrapping if proper
    if proper:
        idx_boot = rng.choice(n, size=n, replace=True)
        X = X[idx_boot, :]
        y = y[idx_boot]

    # Decide if classification or regression
    # check if y is numeric or categorical
    # Heuristic: if y_categories is not None => classification
    # else => assume numeric
    if y_categories is not None or pd.api.types.is_object_dtype(y_dtype):
        # classification
        # If y was string or object, we must encode it
        # sklearn tree requires numeric classes, so let's do factorization:
        unique_labels, y_encoded = np.unique(y, return_inverse=True)
        # Fit the classifier
        model = DecisionTreeClassifier(random_state=rng, **kwargs)
        model.fit(X, y_encoded)

        # Find training leaf assignments
        train_leaves = model.apply(X)
        # For each leaf, store the indices of training samples
        leaf_dict = {}
        for i, leaf_id in enumerate(train_leaves):
            if leaf_id not in leaf_dict:
                leaf_dict[leaf_id] = []
            leaf_dict[leaf_id].append(i)

        # For new data
        predict_leaves = model.apply(Xp)

        syn_res = []
        for leaf_id in predict_leaves:
            # gather training y in that leaf
            if leaf_id not in leaf_dict:
                # If no training samples ended up in that leaf, fallback?
                # Let's do random from entire y or skip
                # fallback approach: random from entire training y
                pick_idx = rng.choice(n, size=1, replace=True)[0]
                syn_res.append(y[pick_idx])
            else:
                donors = [y[idx] for idx in leaf_dict[leaf_id]]
                pick_idx = rng.choice(len(donors), size=1, replace=True)[0]
                syn_res.append(donors[pick_idx])

        syn_res = np.array(syn_res)

        # If original was categorical with y_categories,
        # try to preserve them in the final output
        if y_categories is not None:
            # we return a pd.Categorical
            syn_res = pd.Categorical(syn_res, categories=y_categories)

        return {
            "res": syn_res,
            "fit": {
                "model": model,
                "leaf_dict": leaf_dict,
                "unique_labels": unique_labels,
                "is_classification": True,
            },
        }
    else:
        # regression scenario
        model = DecisionTreeRegressor(random_state=rng, **kwargs)
        model.fit(X, y)

        # gather leaf membership for training data
        train_leaves = model.apply(X)
        leaf_dict = {}
        for i, leaf_id in enumerate(train_leaves):
            if leaf_id not in leaf_dict:
                leaf_dict[leaf_id] = []
            leaf_dict[leaf_id].append(i)

        predict_leaves = model.apply(Xp)

        syn_res = []
        for leaf_id in predict_leaves:
            if leaf_id not in leaf_dict:
                # fallback if leaf has no training data
                pick_idx = rng.choice(n, size=1, replace=True)[0]
                syn_res.append(y[pick_idx])
            else:
                donors = [y[idx] for idx in leaf_dict[leaf_id]]
                pick_idx = rng.choice(len(donors), size=1, replace=True)[0]
                syn_res.append(donors[pick_idx])

        syn_res = np.array(syn_res)

        return {
            "res": syn_res,
            "fit": {
                "model": model,
                "leaf_dict": leaf_dict,
                "is_classification": False,
            },
        }
