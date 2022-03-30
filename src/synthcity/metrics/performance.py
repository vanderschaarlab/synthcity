# stdlib
from typing import Any, Dict

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

# synthcity absolute
import synthcity.logger as log
from synthcity.metrics._utils import evaluate_auc
from synthcity.metrics.core import MetricEvaluator
from synthcity.plugins.models.mlp import MLP


class PerformanceEvaluator(MetricEvaluator):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        return "performance"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def _evaluate_performance_classification(
        self,
        model: Any,
        model_args: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        """
        Evaluate a classification task.

        Returns: AUCROC.
            1 means perfect predictions.
            0 means only incorrect predictions.
        """
        encoder = LabelEncoder().fit(y_train)
        enc_y_train = encoder.transform(y_train)
        if "n_units_out" in model_args:
            model_args["n_units_out"] = len(np.unique(y_train))
        try:
            enc_y_test = encoder.transform(y_test)
            estimator = model(**model_args).fit(X_train, enc_y_train)
            y_pred = estimator.predict_proba(X_test)
            score, _ = evaluate_auc(enc_y_test, y_pred)
        except BaseException as e:
            log.error(f"classifier evaluation failed {e}")
            score = 0

        return score

    def _evaluate_performance_regression(
        self,
        model: Any,
        model_args: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        """
        Evaluate a regression task.

        Returns: 1/ (1 + RMSE).
            0 means perfect predictions.
            The lower the negative value, the bigger the error in the predictions.
        """
        try:
            estimator = model(**model_args).fit(X_train, y_train)
            y_pred = estimator.predict(X_test)

            score = mean_squared_error(y_test, y_pred)
        except BaseException as e:
            log.error(f"regression evaluation failed {e}")
            score = 100

        return 1 / (1 + score)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate_test_performance(
        self,
        clf_model: Any,
        clf_args: Dict,
        regression_model: Any,
        regression_args: Any,
        X_gt: pd.DataFrame,
        X_syn: pd.DataFrame,
    ) -> float:
        """Train a classifier or regressor on the synthetic data and evaluate the performance on real test data. Returns the average performance discrepancy between training on real data vs on synthetic data.

        Score:
            close to 0: similar performance
            1: massive performance degradation
        """

        target_col = X_gt.columns[-1]

        iter_X_gt = X_gt.drop(columns=[target_col])
        iter_y_gt = X_gt[target_col]

        iter_X_syn = X_syn.drop(columns=[target_col])
        iter_y_syn = X_syn[target_col]

        if len(iter_y_gt.unique()) < 5:
            eval_cbk = self._evaluate_performance_classification
            skf = StratifiedKFold(n_splits=3)
            model = clf_model
            model_args = clf_args
        else:
            eval_cbk = self._evaluate_performance_regression
            model = regression_model
            model_args = regression_args
            skf = KFold(n_splits=3)

        real_scores = []
        syn_scores = []
        for train_idx, test_idx in skf.split(iter_X_gt, iter_y_gt):
            train_data = np.asarray(iter_X_gt.loc[train_idx])
            test_data = np.asarray(iter_X_gt.loc[test_idx])
            train_labels = np.asarray(iter_y_gt.loc[train_idx])
            test_labels = np.asarray(iter_y_gt.loc[test_idx])

            real_score = eval_cbk(
                model, model_args, train_data, train_labels, test_data, test_labels
            )
            synth_score = eval_cbk(
                model, model_args, iter_X_syn, iter_y_syn, test_data, test_labels
            )

            real_scores.append(real_score)
            syn_scores.append(synth_score)

        return np.mean(real_scores) - np.mean(syn_scores)


class PerformanceEvaluatorXGB(PerformanceEvaluator):
    """Train an XGBoost classifier or regressor on the synthetic data and evaluate the performance on real test data. Returns the average performance discrepancy between training on real data vs on synthetic data.

    Score:
        close to 0: similar performance
        1: massive performance degradation
    """

    @staticmethod
    def name() -> str:
        return "performance_xgb"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: pd.DataFrame,
        X_syn: pd.DataFrame,
    ) -> float:

        return self._evaluate_test_performance(
            XGBClassifier,
            {
                "n_jobs": 1,
                "verbosity": 0,
                "use_label_encoder": True,
                "depth": 3,
            },
            XGBRegressor,
            {
                "n_jobs": 1,
                "verbosity": 0,
                "use_label_encoder": False,
                "depth": 3,
            },
            X_gt,
            X_syn,
        )


class PerformanceEvaluatorLinear(PerformanceEvaluator):
    """Train a Linear classifier or regressor on the synthetic data and evaluate the performance on real test data. Returns the average performance discrepancy between training on real data vs on synthetic data.

    Score:
        close to 0: similar performance
        1: massive performance degradation
    """

    @staticmethod
    def name() -> str:
        return "performance_linear"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: pd.DataFrame,
        X_syn: pd.DataFrame,
    ) -> float:

        return self._evaluate_test_performance(
            LogisticRegression, {}, LinearRegression, {}, X_gt, X_syn
        )


class PerformanceEvaluatorMLP(PerformanceEvaluator):
    """Train a Neural Net classifier or regressor on the synthetic data and evaluate the performance on real test data. Returns the average performance discrepancy between training on real data vs on synthetic data.

    Score:
        close to 0: similar performance
        1: massive performance degradation
    """

    @staticmethod
    def name() -> str:
        return "performance_mlp"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: pd.DataFrame,
        X_syn: pd.DataFrame,
    ) -> float:

        return self._evaluate_test_performance(
            MLP,
            {
                "task_type": "classification",
                "n_units_in": X_gt.shape[1] - 1,
                "n_units_out": 0,
            },
            MLP,
            {
                "task_type": "regression",
                "n_units_in": X_gt.shape[1] - 1,
                "n_units_out": 1,
            },
            X_gt,
            X_syn,
        )
