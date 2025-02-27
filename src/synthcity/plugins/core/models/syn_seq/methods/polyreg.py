import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def syn_polyreg(y, X, random_state=0, solver="lbfgs", max_iter=200, **kwargs):
    """
    Fit a multinomial logistic regression model for a “polyreg” synthesis.

    For a floating‐point target y:
      - If it has more than 10 unique values, we bin it into 10 discrete bins.
      - Otherwise, we factorize y into discrete labels.
    The bin edges are saved in case you wish to map the predicted bin back.
    """
    y = np.asarray(y)
    X = np.asarray(X)
    
    if np.issubdtype(y.dtype, np.floating):
        unique_values = np.unique(y)
        if len(unique_values) > 10:
            # Bin into 10 bins and save bin edges.
            y_binned, bin_edges = pd.cut(y, bins=10, retbins=True, labels=False)
            y_fit = y_binned.astype(int)
        else:
            # For fewer unique values, factorize to get categorical codes.
            y_fit, _ = pd.factorize(y)
            bin_edges = None
    else:
        y_fit = y
        bin_edges = None
        
    model = LogisticRegression(
        multi_class="multinomial",
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        **kwargs
    )
    model.fit(X, y_fit)
    return {"name": "polyreg", "model": model, "bin_edges": bin_edges, "random_state": random_state}

def generate_polyreg(fitted_model, X_new, random_state=None, **kwargs):
    """
    Generate synthetic values using the fitted polyreg model.

    The fitted model predicts class probabilities for each bin.
    For each sample, a bin is chosen by sampling from the predicted probabilities.
    If binning was used during training, the bin index is mapped back
    to a numeric value by computing the bin centers.
    """
    model = fitted_model["model"]
    bin_edges = fitted_model.get("bin_edges")
    if random_state is None:
        random_state = fitted_model.get("random_state", 0)
    rng = np.random.default_rng(random_state)
    
    # Predict class probabilities
    probs = model.predict_proba(X_new)
    classes = model.classes_
    n_samples = X_new.shape[0]
    y_binned = np.empty(n_samples, dtype=int)
    for i in range(n_samples):
        y_binned[i] = rng.choice(classes, p=probs[i])
        
    if bin_edges is not None:
        # Map the bin index to the bin center.
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        y_generated = bin_centers[y_binned]
    else:
        y_generated = y_binned
    return y_generated
