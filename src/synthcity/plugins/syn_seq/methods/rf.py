# synthcity/plugins/syn_seq/methods/rf.py

"""
Implements a random forest donor-based synthesis approach, analogous (in spirit) 
to R's `syn.rf` from synthpop, using scikit-learn's RandomForestClassifier/RandomForestRegressor.

Key ideas:
    - If y is numeric => use RandomForestRegressor.
    - If y is categorical => use RandomForestClassifier.
    - Each tree can "donate" leaf membership. We find the leaf node that X_i or Xp_j lands in
      for each tree. Then we gather the observed training y in those leaf node(s). 
      We randomly sample from that "donor" pool.

    - The final synthetic y for each row in Xp is drawn from the union of donor pools 
      across all trees. In other words, we collect donors from each tree's leaf node
      for that row, combine them, and sample from the combined set.

    - If proper=True => we bootstrap (X,y) before training. 
    - We also handle the fallback if no donors found (which should be unlikely with many trees).

Usage:
    from rf import syn_rf

    result = syn_rf(
        y, X, Xp, 
        proper=False, 
        n_estimators=10, 
        random_state=0, 
        **kwargs
    )
    y_syn = result["res"]
    print(y_syn)

Notes:
    - This code is not a direct line-for-line R -> Python translation. 
      It captures the main concept: random forest leaf-based sampling 
      from the training data distribution.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def syn_rf(
    y: Union[pd.Series, np.ndarray],
    X: Union[pd.DataFrame, np.ndarray],
    Xp: Union[pd.DataFrame, np.ndarray],
    proper: bool = False,
    random_state: Optional[int] = None,
    n_estimators: int = 10,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Synthetic data generation using a random forest approach.

    Args:
        y: 1D array-like of shape (n,). The response variable.
           - If numeric => random forest regressor.
           - If categorical => random forest classifier.
        X: 2D array-like of shape (n, p). Covariates for training data.
        Xp: 2D array-like of shape (m, p). Covariates for new data to generate synthetic y.
        proper: bool, default=False. If True, bootstrap (X,y) prior to training.
        random_state: Optional[int]. Random seed for reproducibility.
        n_estimators: int, default=10. Number of trees.
        **kwargs: Additional arguments passed to RandomForestRegressor/Classifier, e.g.:
                  - max_depth=None
                  - min_samples_leaf=5
                  - etc.

    Returns:
        A dictionary with:
            "res": array-like of shape (m,) with the synthetic outcome for each row in Xp.
            "fit": dictionary containing fitted forest and additional metadata.
    """

    rng = np.random.RandomState(random_state)

    # Convert y to array, track if it is categorical
    if isinstance(y, pd.Series):
        y_dtype = y.dtype
        if pd.api.types.is_categorical_dtype(y):
            y_categories = y.cat.categories
        else:
            y_categories = None
        y = y.values
    else:
        y_dtype = None
        y_categories = None

    X = np.asarray(X)
    Xp = np.asarray(Xp)
    n = len(y)

    # If proper => bootstrap
    if proper:
        idx_boot = rng.choice(n, size=n, replace=True)
        X = X[idx_boot, :]
        y = y[idx_boot]

    # Decide if classification or regression
    # Heuristic: if we had categories or dtype is object, treat as classification
    if y_categories is not None or pd.api.types.is_object_dtype(y_dtype):
        # classification
        # Factorize if needed
        unique_labels, y_encoded = np.unique(y, return_inverse=True)
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=rng, **kwargs
        )
        rf_model.fit(X, y_encoded)

        # For each tree, store leaf memberships
        # scikit's .apply => shape=(n_samples, n_trees), each entry is the leaf index
        train_leaves = rf_model.apply(X)  # shape = (n, n_trees)
        # Build a structure: for each tree => dict(leaf_id -> list of training indices)
        leaf_map = []
        for t_idx in range(n_estimators):
            leaf_dict = {}
            for i, leaf_id in enumerate(train_leaves[:, t_idx]):
                if leaf_id not in leaf_dict:
                    leaf_dict[leaf_id] = []
                leaf_dict[leaf_id].append(i)
            leaf_map.append(leaf_dict)

        # predict leaves for Xp
        xp_leaves = rf_model.apply(Xp)  # shape = (m, n_trees)

        syn_res = []
        for row_idx in range(len(Xp)):
            # for each row in Xp, gather donors from each tree
            donors_union = []
            for t_idx in range(n_estimators):
                leaf_id = xp_leaves[row_idx, t_idx]
                # gather the training indices in that leaf
                if leaf_id in leaf_map[t_idx]:
                    donors_union.extend(leaf_map[t_idx][leaf_id])
            if len(donors_union) == 0:
                # fallback => random from entire training
                pick_idx = rng.choice(n, size=1)[0]
                chosen_val = y[pick_idx]
            else:
                pick_idx = rng.choice(donors_union, size=1)[0]
                chosen_val = y[pick_idx]
            syn_res.append(chosen_val)

        syn_res = np.array(syn_res)
        # if the original was a known categorical, cast back
        if y_categories is not None:
            syn_res = pd.Categorical(syn_res, categories=y_categories)

        return {
            "res": syn_res,
            "fit": {
                "model": rf_model,
                "leaf_map": leaf_map,
                "is_classification": True,
                "unique_labels": unique_labels,
            },
        }
    else:
        # numeric => regression
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=rng, **kwargs
        )
        rf_model.fit(X, y)

        train_leaves = rf_model.apply(X)  # shape = (n, n_trees)
        leaf_map = []
        for t_idx in range(n_estimators):
            leaf_dict = {}
            for i, leaf_id in enumerate(train_leaves[:, t_idx]):
                if leaf_id not in leaf_dict:
                    leaf_dict[leaf_id] = []
                leaf_dict[leaf_id].append(i)
            leaf_map.append(leaf_dict)

        xp_leaves = rf_model.apply(Xp)

        syn_res = []
        for row_idx in range(len(Xp)):
            donors_union = []
            for t_idx in range(n_estimators):
                leaf_id = xp_leaves[row_idx, t_idx]
                if leaf_id in leaf_map[t_idx]:
                    donors_union.extend(leaf_map[t_idx][leaf_id])
            if len(donors_union) == 0:
                pick_idx = rng.choice(n, size=1)[0]
                chosen_val = y[pick_idx]
            else:
                pick_idx = rng.choice(donors_union, size=1)[0]
                chosen_val = y[pick_idx]
            syn_res.append(chosen_val)

        syn_res = np.array(syn_res)

        return {
            "res": syn_res,
            "fit": {
                "model": rf_model,
                "leaf_map": leaf_map,
                "is_classification": False,
            },
        }
