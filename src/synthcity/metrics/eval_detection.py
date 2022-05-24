# stdlib
from typing import Any, Dict, Tuple

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
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

    def _normalize_covariates(
        self, X_gt_train: pd.DataFrame, X_gt_test: pd.DataFrame, X_syn: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        X_gt_train_norm = X_gt_train.copy()
        X_gt_test_norm = X_gt_test.copy()
        X_syn_norm = X_syn.copy()
        if self._task_type != "survival_analysis":
            X_gt_train_norm = X_gt_train_norm.drop(
                columns=[self._target_column, self._time_to_event_column]
            )
            X_gt_test_norm = X_gt_test_norm.drop(
                columns=[self._target_column, self._time_to_event_column]
            )
            X_syn = X_syn.drop(
                columns=[self._target_column, self._time_to_event_column]
            )

        scaler = MinMaxScaler().fit(X_gt_train_norm)
        return (
            pd.DataFrame(scaler.transform(X_gt_train_norm), columns=X_gt_train.columns),
            pd.DataFrame(scaler.transform(X_gt_test_norm), columns=X_gt_train.columns),
            pd.DataFrame(scaler.transform(X_syn_norm), columns=X_syn.columns),
        )

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

        skf = StratifiedKFold(
            n_splits=self._n_folds, shuffle=True, random_state=self._random_seed
        )
        for train_idx, test_idx in skf.split(data, labels):
            train_data = data.loc[train_idx]
            train_labels = labels.loc[train_idx]
            test_data = data.loc[test_idx]
            test_labels = labels.loc[test_idx]

            model = model_template(**model_args).fit(
                train_data.values.astype(float), train_labels.values
            )

            test_pred = model.predict(test_data.values.astype(float))

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
    def evaluate(
        self, X_gt_train: pd.DataFrame, X_gt_test: pd.DataFrame, X_syn: pd.DataFrame
    ) -> Dict:
        X_gt_train, X_gt_test, X_syn = self._normalize_covariates(
            X_gt_train, X_gt_test, X_syn
        )
        model_template = XGBClassifier
        model_args = {
            "n_jobs": -1,
            "verbosity": 0,
            "use_label_encoder": False,
            "depth": 3,
            "random_state": self._random_seed,
        }

        return self._evaluate_detection(model_template, X_gt_train, X_syn, **model_args)


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
    def evaluate(
        self, X_gt_train: pd.DataFrame, X_gt_test: pd.DataFrame, X_syn: pd.DataFrame
    ) -> Dict:
        X_gt_train, X_gt_test, X_syn = self._normalize_covariates(
            X_gt_train, X_gt_test, X_syn
        )
        model_args = {
            "task_type": "classification",
            "n_units_in": X_gt_train.shape[1],
            "n_units_out": 2,
            "seed": self._random_seed,
        }
        return self._evaluate_detection(
            MLP,
            X_gt_train,
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
        X_gt_train: pd.DataFrame,
        X_gt_test: pd.DataFrame,
        X_syn: pd.DataFrame,
    ) -> Dict:
        X_gt_train, X_gt_test, X_syn = self._normalize_covariates(
            X_gt_train, X_gt_test, X_syn
        )
        scores = []

        for component in [1, 5, 10]:
            gmm = GaussianMixture(n_components=component, covariance_type="diag")
            gmm.fit(X_gt_train)

            scores.append(gmm.score(X_syn))  # Higher is better

        scores_np = np.asarray(scores)
        scores_np = (scores_np - np.min(scores_np)) / (
            np.max(scores_np) - np.min(scores_np)
        )  # transform scores to [0, 1]
        scores_np = 1 - scores_np  # invert scores - lower is better

        return {self._reduction: self.reduction()(scores_np)}
