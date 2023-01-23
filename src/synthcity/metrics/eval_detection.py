# stdlib
import platform
from typing import Any, Dict

# third party
import numpy as np
from pydantic import validate_arguments
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# synthcity absolute
import synthcity.logger as log
from synthcity.metrics.core import MetricEvaluator
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.models.mlp import MLP
from synthcity.utils.serialization import load_from_file, save_to_file


class DetectionEvaluator(MetricEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_detection.DetectionEvaluator
        :parts: 1


    Train a SKLearn classifier to detect the synthetic data from real data.

    Synthetic and real data are combined to form a new dataset.
    K-fold cross validation is performed to see how well a classifier can distinguish real from synthetic.

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
        X_gt: DataLoader,
        X_syn: DataLoader,
        **model_args: Any,
    ) -> Dict:
        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{X_gt.hash()}_{X_syn.hash()}_{self._reduction}_{platform.python_version()}.bkp"
        )
        if self.use_cache(cache_file):
            results = load_from_file(cache_file)
            log.info(f" Detection eval for {self.name()} : {results}")
            return results

        arr_gt = X_gt.numpy()
        labels_gt = np.asarray([0] * len(X_gt))

        arr_syn = X_syn.numpy()
        labels_syn = np.asarray([1] * len(X_syn))

        data = np.concatenate([arr_gt, arr_syn])
        labels = np.concatenate([labels_gt, labels_syn])

        res = []

        skf = StratifiedKFold(
            n_splits=self._n_folds, shuffle=True, random_state=self._random_state
        )
        for train_idx, test_idx in skf.split(data, labels):
            train_data = data[train_idx]
            train_labels = labels[train_idx]
            test_data = data[test_idx]
            test_labels = labels[test_idx]

            model = model_template(**model_args).fit(
                train_data.astype(float), train_labels
            )

            test_pred = model.predict_proba(test_data.astype(float))[:, 1]

            score = roc_auc_score(test_labels, test_pred)
            res.append(score)

        results = {self._reduction: float(self.reduction()(res))}
        log.info(f" Detection eval for {self.name()} : {results}")

        save_to_file(cache_file, results)

        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_default(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> float:
        return self.evaluate(X_gt, X_syn)[self._reduction]


class SyntheticDetectionXGB(DetectionEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_detection.SyntheticDetectionXGB
        :parts: 1

    Train a XGBoostclassifier to detect the synthetic data.

    Returns:
        The average AUCROC score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "detection_xgb"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        model_template = XGBClassifier
        model_args = {
            "n_jobs": 2,
            "verbosity": 0,
            "depth": 3,
            "random_state": self._random_state,
        }

        return self._evaluate_detection(model_template, X_gt, X_syn, **model_args)


class SyntheticDetectionMLP(DetectionEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_detection.SyntheticDetectionMLP
        :parts: 1

    Train a MLP classifier to detect the synthetic data.

    Returns:
        The average AUCROC score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "detection_mlp"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        model_args = {
            "task_type": "classification",
            "n_units_in": X_gt.shape[1],
            "n_units_out": 2,
            "random_state": self._random_state,
        }
        return self._evaluate_detection(
            MLP,
            X_gt,
            X_syn,
            **model_args,
        )


class SyntheticDetectionLinear(DetectionEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_detection.SyntheticDetectionLinear
        :parts: 1

    Train a LogisticRegression classifier to detect the synthetic data.

    Returns:
        The average AUCROC score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "detection_linear"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        model_args = {
            "random_state": self._random_state,
            "n_jobs": -1,
            "max_iter": 10000,
        }
        return self._evaluate_detection(
            LogisticRegression,
            X_gt,
            X_syn,
            **model_args,
        )


class SyntheticDetectionGMM(DetectionEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_detection.SyntheticDetectionGMM
        :parts: 1

    Train a GaussianMixture model to detect synthetic data.

    Returns:
        The average score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "detection_gmm"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        model_args = {
            "n_components": min(10, len(X_gt)),
            "random_state": self._random_state,
        }
        return self._evaluate_detection(
            GaussianMixture,
            X_gt,
            X_syn,
            **model_args,
        )
