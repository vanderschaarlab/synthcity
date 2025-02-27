import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def syn_logreg(y, X, random_state=0, is_classification=True, **kwargs):
    """
    Fit a logistic regression model.
    
    If is_classification is True and y is of a floating‐point type,
    we convert y to discrete values (using categorical codes) so that
    LogisticRegression does not complain about a continuous target.
    
    If is_classification is False, we use LinearRegression instead.
    """
    y = np.asarray(y)
    X = np.asarray(X)
    
    if is_classification:
        # If y is a float type (i.e. continuous), convert to integer codes.
        if np.issubdtype(y.dtype, np.floating):
            # Using pd.Categorical will map the unique values to integer codes.
            y = pd.Categorical(y).codes
        else:
            y = y.astype(int)
        model = LogisticRegression(random_state=random_state, **kwargs)
        model.fit(X, y)
        return {"name": "logreg", "model": model, "random_state": random_state}
    else:
        # For regression tasks, use LinearRegression.
        from sklearn.linear_model import LinearRegression
        model = LinearRegression(**kwargs)
        model.fit(X, y)
        return {"name": "logreg", "model": model, "random_state": random_state}

def generate_logreg(fitted_model, X_new, random_state=None, **kwargs):
    """
    Generate predictions using the fitted logistic regression model.
    
    For classification, simply use the predict method.
    """
    model = fitted_model["model"]
    y_pred = model.predict(X_new)
    return y_pred
