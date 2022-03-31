# stdlib
from typing import Any, Dict

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# synthcity absolute
from synthcity.metrics.core import MetricEvaluator
from synthcity.plugins.models.mlp import MLP


class DetectionEvaluator(MetricEvaluator):
    """Train a SKLearn classifier to detect the synthetic data.

    Returns:
        The average AUCROC score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        return "detection"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate_detection(
        self,
        model_template: Any,
        X_gt: pd.DataFrame,
        X_syn: pd.DataFrame,
        **model_args: Any,
    ) -> Dict:
        X_gt = X_gt.reset_index(drop=True)
        labels_gt = pd.Series([0] * len(X_gt))

        X_syn = X_syn.reset_index(drop=True)
        labels_syn = pd.Series([1] * len(X_syn))

        data = pd.concat([X_gt, X_syn]).reset_index(drop=True)
        labels = pd.concat([labels_gt, labels_syn]).reset_index(drop=True)

        res = []

        skf = StratifiedKFold(n_splits=3)
        for train_idx, test_idx in skf.split(data, labels):
            train_data = data.loc[train_idx]
            train_labels = labels.loc[train_idx]
            test_data = data.loc[test_idx]
            test_labels = labels.loc[test_idx]

            model = model_template(**model_args).fit(
                train_data.values, train_labels.values
            )

            test_pred = model.predict(test_data.values)

            score = roc_auc_score(np.asarray(test_labels), np.asarray(test_pred))
            res.append(score)

        return {self._reduction: float(self.reduction()(res))}


class SyntheticDetectionXGB(DetectionEvaluator):
    """Train a XGBoostclassifier to detect the synthetic data.

    Returns:
        The average AUCROC score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    @staticmethod
    def name() -> str:
        return "detection_xgb"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: pd.DataFrame, X_syn: pd.DataFrame) -> Dict:
        model_template = XGBClassifier
        model_args = {
            "n_jobs": 1,
            "verbosity": 0,
            "use_label_encoder": False,
            "depth": 3,
        }

        return self._evaluate_detection(model_template, X_gt, X_syn, **model_args)


class SyntheticDetectionMLP(DetectionEvaluator):
    """Train a MLP classifier to detect the synthetic data.

    Returns:
        The average AUCROC score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    @staticmethod
    def name() -> str:
        return "detection_mlp"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: pd.DataFrame, X_syn: pd.DataFrame) -> Dict:
        model_args = {
            "task_type": "classification",
            "n_units_in": X_gt.shape[1],
            "n_units_out": 2,
        }
        return self._evaluate_detection(
            MLP,
            X_gt,
            X_syn,
            **model_args,
        )


class SyntheticDetectionGMM(DetectionEvaluator):
    """Train a GaussianMixture model to detect synthetic data.

    Returns:
        The average score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    @staticmethod
    def name() -> str:
        return "detection_gmm"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: pd.DataFrame,
        X_syn: pd.DataFrame,
    ) -> Dict:

        scores = []

        for component in [1, 5, 10]:
            gmm = GaussianMixture(n_components=component, covariance_type="diag")
            gmm.fit(X_gt)

            scores.append(gmm.score(X_syn))  # Higher is better

        scores_np = np.asarray(scores)
        scores_np = (scores_np - np.min(scores_np)) / (
            np.max(scores_np) - np.min(scores_np)
        )  # transform scores to [0, 1]
        scores_np = 1 - scores_np  # invert scores - lower is better

        return {self._reduction: self.reduction()(scores_np)}
