# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def detect_synth(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_synth: pd.DataFrame, y_synth: pd.Series
) -> float:
    """Train a classifier to detect synthetic data.

    Returns:
        The average AUCROC score for detecting synthtic data. 1 means the synthetic and real data are totally distinguishable.
        Lower is better.
    """

    X_gt["target"] = y_gt
    X_synth["target"] = y_synth

    X_gt = X_gt.reset_index(drop=True)
    labels_gt = pd.Series([0] * len(X_gt))

    X_synth = X_synth.reset_index(drop=True)
    labels_synth = pd.Series([1] * len(X_synth))

    data = X_gt.append(X_synth).reset_index(drop=True)
    labels = labels_gt.append(labels_synth).reset_index(drop=True)

    res = []

    skf = StratifiedKFold(n_splits=3)
    for train_idx, test_idx in skf.split(data, labels):
        train_data = data.loc[train_idx]
        train_labels = labels.loc[train_idx]
        test_data = data.loc[test_idx]
        test_labels = labels.loc[test_idx]

        model = XGBClassifier(verbosity=0, use_label_encoder=False).fit(
            train_data, train_labels
        )

        test_pred = model.predict(test_data)

        score = roc_auc_score(test_labels, test_pred)
        res.append(score)

    return np.mean(res)
