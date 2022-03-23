# stdlib
from typing import Any

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# synthcity absolute
from synthcity.plugins.models.mlp import MLP


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_sklearn_detection_synthetic(
    model_template: Any,
    X_gt: pd.DataFrame,
    y_gt: pd.Series,
    X_syn: pd.DataFrame,
    y_syn: pd.Series,
    **model_args: Any,
) -> float:
    """Train a SKLearn classifier to detect the synthetic data.

    Returns:
        The average AUCROC score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    X_gt["target"] = y_gt
    X_syn["target"] = y_syn

    X_gt = X_gt.reset_index(drop=True)
    labels_gt = pd.Series([0] * len(X_gt))

    X_syn = X_syn.reset_index(drop=True)
    labels_syn = pd.Series([1] * len(X_syn))

    data = X_gt.append(X_syn).reset_index(drop=True)
    labels = labels_gt.append(labels_syn).reset_index(drop=True)

    res = []

    skf = StratifiedKFold(n_splits=3)
    for train_idx, test_idx in skf.split(data, labels):
        train_data = data.loc[train_idx]
        train_labels = labels.loc[train_idx]
        test_data = data.loc[test_idx]
        test_labels = labels.loc[test_idx]

        model = model_template(**model_args).fit(train_data.values, train_labels.values)

        test_pred = model.predict(test_data.values)

        score = roc_auc_score(np.asarray(test_labels), np.asarray(test_pred))
        res.append(score)

    return np.mean(res)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_xgb_detection_synthetic(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_syn: pd.DataFrame, y_syn: pd.Series
) -> float:
    """Train a XGBoostclassifier to detect the synthetic data.

    Returns:
        The average AUCROC score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    model_template = XGBClassifier
    model_args = {
        "n_jobs": 1,
        "verbosity": 0,
        "use_label_encoder": False,
        "depth": 3,
    }

    return evaluate_sklearn_detection_synthetic(
        model_template, X_gt, y_gt, X_syn, y_syn, **model_args
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_mlp_detection_synthetic(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_syn: pd.DataFrame, y_syn: pd.Series
) -> float:
    """Train a MLP classifier to detect the synthetic data.

    Returns:
        The average AUCROC score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    model_args = {
        "task_type": "classification",
    }
    return evaluate_sklearn_detection_synthetic(
        MLP,
        X_gt,
        y_gt,
        X_syn,
        y_syn,
        **model_args,
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_gmm_detection_synthetic(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_syn: pd.DataFrame, y_syn: pd.Series
) -> float:
    """Train a GaussianMixture model to detect synthetic data.

    Returns:
        The average score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    scores = []

    X_gt["target"] = y_gt
    X_syn["target"] = y_syn

    for component in [1, 5, 10]:
        gmm = GaussianMixture(n_components=component, covariance_type="diag")
        gmm.fit(X_gt)

        scores.append(gmm.score(X_syn))  # Higher is better

    scores_np = np.asarray(scores)
    scores_np = (scores_np - np.min(scores_np)) / (
        np.max(scores_np) - np.min(scores_np)
    )  # transform scores to [0, 1]
    scores_np = 1 - scores_np  # invert scores - lower is better

    return np.mean(scores_np)
