# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# synthcity absolute
from synthcity.metrics._utils import evaluate_auc


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def performance_train_synth_classifier(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_synth: pd.DataFrame, y_synth: pd.Series
) -> float:
    """Train a classifier on synthetic data and evaluate the performance on real test data.

    Returns:
        The average AUCROC score for detecting synthtic data. 1 means the synthetic and real data are totally distinguishable.
        Lower is better.
    """

    model = XGBClassifier
    model_args = {
        "verbosity": 0,
        "use_label_encoder": False,
        "depth": 3,
    }

    res = []
    skf = StratifiedKFold(n_splits=3)
    for train_idx, test_idx in skf.split(X_gt, y_gt):
        train_data = X_gt.loc[train_idx]
        test_data = X_gt.loc[test_idx]
        train_labels = y_gt.loc[train_idx]
        test_labels = y_gt.loc[test_idx]

        try:
            real_model = model(**model_args).fit(train_data, train_labels)
            test_pred = real_model.predict_proba(test_data)
            real_score, _ = evaluate_auc(test_labels, test_pred)

        except BaseException:
            real_score = 0

        try:
            synth_model = model(**model_args).fit(X_synth, y_synth)
            test_pred = synth_model.predict_proba(test_data)
            synth_score, _ = evaluate_auc(test_labels, test_pred)
        except BaseException:
            synth_score = 0

        res.append(real_score - synth_score)

    return np.mean(res)
