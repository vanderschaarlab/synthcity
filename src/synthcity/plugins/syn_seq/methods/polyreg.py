# synthcity/plugins/syn_seq/methods/polyreg.py

"""
Implements a polytomous (multinomial) regression approach, analogous to R's syn.polyreg in synthpop.

Key ideas from R syn.polyreg:
    - For a categorical response variable (with k>2 possible categories),
      fit a multinomial model using e.g. 'multinom' from nnet in R.
    - Predict probabilities for new data Xp.
    - For each row in Xp, draw a random category from the predicted probabilities.

In this Python version:
    - We use scikit-learn's LogisticRegression with multi_class="multinomial" as the default approach.
    - The user can specify "proper=True" to perform bootstrapping on (X, y) before model fitting.
    - If y is factor/categorical, we keep track of its categories so we can return consistent labels.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def syn_polyreg(
    y: Union[pd.Series, np.ndarray],
    X: Union[pd.DataFrame, np.ndarray],
    Xp: Union[pd.DataFrame, np.ndarray],
    proper: bool = False,
    random_state: Optional[int] = None,
    max_iter: int = 1000,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Synthesis for categorical/multinomial response variables by a polytomous regression model.

    Args:
        y: 1D array-like. The response variable, containing multiple categories.
            If given as pandas Series with categorical dtype, we preserve categories.
        X: 2D array-like of shape (n, p). Covariates of the training data.
        Xp: 2D array-like of shape (m, p). Covariates of the new data to synthesize for.
        proper: bool, default=False. If True, apply bootstrapping to (X, y) before fitting the model.
        random_state: Optional[int]. Seed for reproducibility.
        max_iter: int, default=1000. Max iteration for the underlying solver.
        **kwargs: Additional arguments passed to LogisticRegression constructor.
                  e.g. solver="lbfgs", C=1.0, etc.

    Returns:
        A dictionary with:
            "res": array of shape (m,) of synthetic categories for each row in Xp.
            "fit": a dictionary describing the fitted model + metadata.
    """
    rng = np.random.RandomState(random_state)

    # Convert y to numpy array if not already
    # Try to preserve categories if y is a pandas Series with categorical dtype.
    if isinstance(y, pd.Series):
        y_categories = y.cat.categories if pd.api.types.is_categorical_dtype(y) else None
        y = y.values
    else:
        y_categories = None

    y = np.asarray(y)
    X = np.asarray(X)
    Xp = np.asarray(Xp)

    n = len(y)
    # Bootstrapping if proper
    if proper:
        idx_boot = rng.choice(n, size=n, replace=True)
        X = X[idx_boot, :]
        y = y[idx_boot]

    # Fit a multinomial logistic regression
    # By default, scikit-learn's LogisticRegression expects numeric y for classification,
    # but we can encode categories with e.g. y.astype("int") if y is already numeric-coded.
    # If y has string labels, it should still work as scikit-learn will encode them internally.
    model_params = {
        "multi_class": "multinomial",
        "max_iter": max_iter,
        # common solver for multinomial is lbfgs, but user can override via kwargs
        "solver": kwargs.pop("solver", "lbfgs"),
        "random_state": rng,
    }
    model_params.update(kwargs)
    model = LogisticRegression(**model_params)

    # Fit
    model.fit(X, y)

    # Predicted probabilities for Xp
    prob = model.predict_proba(Xp)

    # Draw from predicted distribution
    syn_labels = []
    for row_p in prob:
        # sample from the row_p distribution
        cat_idx = rng.choice(len(row_p), p=row_p)
        syn_labels.append(model.classes_[cat_idx])

    syn_labels = np.asarray(syn_labels)

    # If original was a categorical type, we try to map back to those categories
    if y_categories is not None and len(model.classes_) == len(y_categories):
        # We assume model.classes_ is in the same order as y_categories
        # If not, or if the numeric codes differ, further mapping might be needed.
        # For now, assume same ordering:
        syn_labels = pd.Categorical.from_codes(
            codes=[
                np.where(model.classes_ == lbl)[0][0]
                for lbl in syn_labels
            ],
            categories=y_categories
        )

    return {
        "res": syn_labels,
        "fit": {
            "model": model,
            "classes_": model.classes_,
            "y_categories": y_categories,
        },
    }
