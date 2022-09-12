# stdlib
import copy
from typing import Any, Dict, Tuple

# third party
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold


def evaluate_classifier(
    estimator: Any,
    X: pd.DataFrame,
    Y: pd.Series,
    n_folds: int = 3,
    seed: int = 0,
) -> Dict:
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)

    metric = "aucroc"
    metric_ = np.zeros(n_folds)

    indx = 0
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_index, test_index in skf.split(X, Y):

        X_train = X.loc[X.index[train_index]]
        Y_train = Y.loc[Y.index[train_index]]
        X_test = X.loc[X.index[test_index]]
        Y_test = Y.loc[Y.index[test_index]]

        model = copy.deepcopy(estimator)
        model.fit(X_train, Y_train)

        preds = model.predict(X_test)

        metric_[indx] = roc_auc_score(Y_test, preds)

        indx += 1

    output_clf = generate_score(metric_)

    return {
        "clf": {
            metric: output_clf,
        },
        "str": {
            metric: print_score(output_clf),
        },
    }


def evaluate_regression(
    estimator: Any,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    n_folds: int = 3,
    seed: int = 0,
    *args: Any,
    **kwargs: Any,
) -> Dict:
    """Helper for evaluating regression tasks.
    Args:
        estimator:
            The regressor to evaluate
        X:
            covariates
        Y:
            outcomes
        n_folds: int
            Number of cross-validation folds
        metric: str
            r2
        seed: int
            Random seed
    """
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)

    metric = "r2"
    metric_ = np.zeros(n_folds)

    indx = 0
    skf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_index, test_index in skf.split(X, Y):

        X_train = X.loc[X.index[train_index]]
        Y_train = Y.loc[Y.index[train_index]]
        X_test = X.loc[X.index[test_index]]
        Y_test = Y.loc[Y.index[test_index]]

        model = copy.deepcopy(estimator)
        model.fit(X_train, Y_train)

        preds = model.predict(X_test)

        metric_[indx] = r2_score(Y_test, preds)

        indx += 1

    output_clf = generate_score(metric_)

    return {
        "clf": {
            metric: output_clf,
        },
        "str": {
            metric: print_score(output_clf),
        },
    }


def generate_score(metric: np.ndarray) -> Tuple[float, float]:
    percentile_val = 1.96
    return (np.mean(metric), percentile_val * np.std(metric) / np.sqrt(len(metric)))


def print_score(score: Tuple[float, float]) -> str:
    return str(round(score[0], 3)) + " +/- " + str(round(score[1], 3))
