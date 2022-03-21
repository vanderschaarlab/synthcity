# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

# synthcity absolute
import synthcity.logger as log
from synthcity.metrics._utils import evaluate_auc


def evaluate_performance_classification(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate a classification task.

    Returns: AUCROC.
        1 means perfect predictions.
        0 means only incorrect predictions.
    """
    model = XGBClassifier
    model_args = {
        "verbosity": 0,
        "use_label_encoder": False,
        "depth": 3,
    }
    try:
        estimator = model(**model_args).fit(X_train, y_train)
        y_pred = estimator.predict_proba(X_test)
        score, _ = evaluate_auc(y_test, y_pred)
    except BaseException as e:
        log.error(f"classifier evaluation failed {e}")
        score = 0

    return score


def evaluate_performance_regression(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate a regression task.

    Returns: -RMSE.
        0 means perfect predictions.
        The lower the negative value, the bigger the error in the predictions.
    """
    model = XGBRegressor
    model_args = {
        "verbosity": 0,
        "use_label_encoder": False,
        "depth": 3,
    }
    try:
        estimator = model(**model_args).fit(X_train, y_train)
        y_pred = estimator.predict(X_test)

        score = mean_squared_error(y_test, y_pred)
    except BaseException as e:
        log.error(f"regression evaluation failed {e}")
        score = 100

    return 1 / (1 + score)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_test_performance(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_synth: pd.DataFrame, y_synth: pd.Series
) -> float:
    """Train a classifier or regressor on the synthetic data and evaluate the performance on real test data. Returns the average performance discrepancy between training on real data vs on synthetic data.

    Score:
        close to 0: similar performance
        1: massive performance degradation
    """

    if len(y_gt.unique()) < 5:
        eval_cbk = evaluate_performance_classification
        skf = StratifiedKFold(n_splits=3)
    else:
        eval_cbk = evaluate_performance_regression
        skf = KFold(n_splits=3)

    real_scores = []
    syn_scores = []
    for train_idx, test_idx in skf.split(X_gt, y_gt):
        train_data = X_gt.loc[train_idx]
        test_data = X_gt.loc[test_idx]
        train_labels = y_gt.loc[train_idx]
        test_labels = y_gt.loc[test_idx]

        real_score = eval_cbk(train_data, train_labels, test_data, test_labels)
        synth_score = eval_cbk(X_synth, y_synth, test_data, test_labels)

        real_scores.append(real_score)
        syn_scores.append(synth_score)

    return np.mean(real_scores) - np.mean(syn_scores)
