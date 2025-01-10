# synthcity/plugins/syn_seq/methods/ctree.py

"""
Implements a conditional inference tree approach, analogous (in spirit) to R's 'ctree' in the party/partykit package.
However, Python does not have a built-in `ctree` identical to R's version. We approximate it with sklearn's
DecisionTreeClassifier/Regressor. The core idea is to split the data based on statistical criteria that mimic 
conditional inference splitting, then use the leaf 'donor pools' to sample from the observed outcomes.

Key ideas from R's syn.ctree:
    1. If y is numeric => use a regression tree (DecisionTreeRegressor).
       - For each new row Xp_i, we find its leaf node, gather the observed y from that leaf, 
         and sample one at random (possibly with some smoothing or proper approach).
    2. If y is categorical => use a classification tree (DecisionTreeClassifier).
       - For each new row Xp_i, find its leaf node, gather the observed y from that leaf, 
         and sample one at random in proportion to that leaf’s distribution of y.
    3. If proper=True => bootstrap (X, y) before fitting, as in some 'proper' multiple imputation logic.
    4. min_samples_leaf, max_depth, etc., can be set as kwargs to control the tree size or structure.
    5. After training, for each row in Xp, we get the leaf node via `.apply(Xp)`, gather the original training y 
       in that leaf node, and randomly sample from them to produce synthetic y. 
       If no donors exist (rare corner case), we fallback to sampling from the entire y.

We call this function `syn_ctree(...)`. The function returns a dict with:
    "res" => the synthetic output array
    "fit" => a dict storing the trained model, leaf map, etc., for debugging or reuse.
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
            - If factor/categorical => classification (DecisionTreeClassifier).
        X: 2D array-like of shape (n, p). Covariates from the training data.
        Xp: 2D array-like of shape (m, p). Covariates for the new data to synthesize.
        proper: bool, default=False. If True, bootstrap (X, y) before fitting 
                to implement a "proper" randomization akin to multiple imputation.
        random_state: Optional[int]. Seed for reproducibility.
        **kwargs: Additional arguments for DecisionTreeRegressor/Classifier, e.g.:
                  - min_samples_leaf=5
                  - max_depth=None
                  - etc.

    Returns:
        A dictionary:
            {
                "res": array-like of shape (m,) with the synthetic y for each row in Xp,
                "fit": {
                    "model": the trained sklearn tree model,
                    "leaf_dict": a dict mapping leaf_id -> list of training sample indices,
                    "unique_labels": optional array of unique labels if classification,
                    "is_classification": bool,
                }
            }
    """

    rng = np.random.RandomState(random_state)

    # Convert inputs to numpy arrays if they are pandas
    if isinstance(y, pd.Series):
        # keep track if it's categorical
        if pd.api.types.is_categorical_dtype(y):
            y_categories = y.cat.categories
        else:
            y_categories = None
        y = y.values
    else:
        y_categories = None

    X = np.asarray(X)
    Xp = np.asarray(Xp)

    n = len(y)

    # If proper => bootstrap (X, y)
    if proper:
        idx_boot = rng.choice(n, size=n, replace=True)
        X = X[idx_boot, :]
        y = y[idx_boot]

    # Determine classification vs. regression
    # If the original y had categories (y_categories), or if dtype is object/strings => classification.
    is_classification = False
    if y_categories is not None:
        is_classification = True
    elif pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y):
        is_classification = True

    # If still not certain, try numeric checks
    if not is_classification:
        # numeric if all values are number
        # we assume an integer/float means regression
        # if there's a small number of unique values, user might want classification, 
        # but we'll keep it simple
        pass

    if is_classification:
        # Encode strings if needed
        unique_labels, y_encoded = np.unique(y, return_inverse=True)
        model = DecisionTreeClassifier(random_state=rng, **kwargs)
        model.fit(X, y_encoded)

        train_leaves = model.apply(X)
        leaf_dict = {}
        for i, leaf_id in enumerate(train_leaves):
            if leaf_id not in leaf_dict:
                leaf_dict[leaf_id] = []
            leaf_dict[leaf_id].append(i)

        xp_leaves = model.apply(Xp)
        syn_res = []
        for leaf_id in xp_leaves:
            # gather all training y in that leaf
            if leaf_id not in leaf_dict:
                # fallback => random from entire training
                pick_idx = rng.choice(n, size=1)[0]
                syn_res.append(y[pick_idx])
            else:
                donors_idx = leaf_dict[leaf_id]
                pick_idx = rng.choice(donors_idx, size=1)[0]
                syn_res.append(y[pick_idx])

        syn_res = np.array(syn_res)
        # if y_categories is not None, return them with the original categories
        # else keep them as the same dtype
        if y_categories is not None:
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
        # regression
        model = DecisionTreeRegressor(random_state=rng, **kwargs)
        model.fit(X, y)

        train_leaves = model.apply(X)
        leaf_dict = {}
        for i, leaf_id in enumerate(train_leaves):
            if leaf_id not in leaf_dict:
                leaf_dict[leaf_id] = []
            leaf_dict[leaf_id].append(i)

        xp_leaves = model.apply(Xp)
        syn_res = []
        for leaf_id in xp_leaves:
            if leaf_id not in leaf_dict:
                # fallback => random from entire training
                pick_idx = rng.choice(n, size=1)[0]
                syn_res.append(y[pick_idx])
            else:
                donors_idx = leaf_dict[leaf_id]
                pick_idx = rng.choice(donors_idx, size=1)[0]
                syn_res.append(y[pick_idx])

        syn_res = np.array(syn_res, dtype=float)

        return {
            "res": syn_res,
            "fit": {
                "model": model,
                "leaf_dict": leaf_dict,
                "is_classification": False,
            },
        }
