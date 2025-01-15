# synthcity/plugins/syn_seq/methods/polyreg.py

"""
Implements a multinomial (polytomous) regression approach for categorical targets.

Steps:
  1. Identify the categories of the target y. If the target is a pandas Series with a categorical dtype, we preserve those categories.
  2. If `proper=True`, bootstrap (X, y) before fitting.
  3. Fit a scikit-learn LogisticRegression with multi_class="multinomial".
  4. Predict probabilities for each row in Xp, then sample a category from the predicted probability distribution.
  5. Return a dictionary with "res" (the synthetic outcomes) and "fit" (the fitted model and metadata).
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
    Synthesize a categorical (multinomial) variable using multinomial logistic regression.

    Args:
        y: The target array of shape (n,). Contains the observed categories.
        X: Predictor matrix of shape (n, p).
        Xp: Predictor matrix for new data, shape (m, p).
        proper: If True, apply bootstrap to (X, y) before fitting.
        random_state: Random seed for reproducibility.
        max_iter: Max iterations for LogisticRegression.
        **kwargs: Additional arguments for LogisticRegression, e.g. solver="lbfgs", C=1.0, etc.

    Returns:
        A dict with:
          "res": np.ndarray or pd.Categorical of length m (synthetic categories for each row in Xp).
          "fit": a dictionary containing:
              - "model": the fitted LogisticRegression
              - "classes_": model.classes_
              - "y_categories": original category info (if y was categorical)
    """
    rng = np.random.RandomState(random_state)

    # If y is a pandas Series with categorical dtype, keep track of the categories
    y_categories = None
    if isinstance(y, pd.Series):
        if pd.api.types.is_categorical_dtype(y):
            y_categories = y.cat.categories
        y = y.values

    # Convert input data to NumPy arrays if needed
    X = np.asarray(X)
    Xp = np.asarray(Xp)

    n = len(y)
    # Apply bootstrap if proper
    if proper:
        idx_boot = rng.choice(n, size=n, replace=True)
        X = X[idx_boot, :]
        y = y[idx_boot]

    # Prepare arguments for LogisticRegression (multinomial)
    model_params = {
        "multi_class": "multinomial",
        "max_iter": max_iter,
        # If user doesn't specify a solver, default to "lbfgs".
        "solver": kwargs.pop("solver", "lbfgs"),
        "random_state": rng,
    }
    model_params.update(kwargs)

    # Fit the multinomial logistic regression
    model = LogisticRegression(**model_params)
    model.fit(X, y)

    # Predict probabilities for new data
    prob = model.predict_proba(Xp)  # shape = (m, k)

    # Sample from the predicted probability distribution
    syn_labels = []
    for row_p in prob:
        cat_idx = rng.choice(len(row_p), p=row_p)
        syn_labels.append(model.classes_[cat_idx])

    syn_labels = np.asarray(syn_labels)

    # If original y was categorical, try to map back to those categories
    if y_categories is not None and len(model.classes_) == len(y_categories):
        # Align the predicted labels with the original category ordering if possible
        # We create a Series, factor out the codes, then build a pd.Categorical
        df_temp = pd.DataFrame({"pred": syn_labels})
        # Factor to match model.classes_ => integer codes
        df_temp["code"] = df_temp["pred"].apply(
            lambda v: np.where(model.classes_ == v)[0][0]
        )
        # Then map those codes to y_categories
        syn_categorical = pd.Categorical.from_codes(
            codes=df_temp["code"],
            categories=y_categories
        )
        syn_labels = syn_categorical

    return {
        "res": syn_labels,
        "fit": {
            "model": model,
            "classes_": model.classes_,
            "y_categories": y_categories,
        },
    }
