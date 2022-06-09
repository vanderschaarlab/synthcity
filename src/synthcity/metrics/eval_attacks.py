# stdlib
from typing import Any, Dict

# third party
import numpy as np
from pydantic import validate_arguments
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

# synthcity absolute
from synthcity.metrics.core import MetricEvaluator
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.models.mlp import MLP


class AttackEvaluator(MetricEvaluator):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        return "attack"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate_leakage(
        self,
        classifier_template: Any,
        classifier_args: Dict,
        regressor_template: Any,
        regressor_args: Dict,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        if len(X_gt.sensitive_features) == 0:
            return {}

        output = []
        for col in X_gt.sensitive_features:
            if col not in X_syn.columns:
                continue

            target = X_syn[col]
            keys_data = X_syn.drop(columns=[col])

            if len(target.unique()) < 15:
                task_type = "classification"
                encoder = LabelEncoder()
                target = encoder.fit_transform(target)
                if "n_units_out" in classifier_args:
                    classifier_args["n_units_out"] = len(np.unique(target))
                model = classifier_template(**classifier_args)
            else:
                task_type = "regression"
                model = regressor_template(**regressor_args)

            model.fit(keys_data.values, np.asarray(target))

            test_target = X_gt[col]
            if task_type == "classification":
                test_target = encoder.transform(test_target)

            test_keys_data = X_gt.drop(columns=[col])

            preds = model.predict(test_keys_data.values)

            output.append(
                (np.asarray(preds) == np.asarray(test_target)).sum() / (len(preds) + 1)
            )

        if len(output) == 0:
            return {}

        return {self._reduction: self.reduction()(output)}


class DataLeakageMLP(AttackEvaluator):
    @staticmethod
    def name() -> str:
        return "data_leakage_mlp"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        return self._evaluate_leakage(
            MLP,
            {
                "task_type": "classification",
                "n_units_in": X_gt.shape[1] - 1,
                "n_units_out": 0,
                "seed": self._random_seed,
            },
            MLP,
            {
                "task_type": "regression",
                "n_units_in": X_gt.shape[1] - 1,
                "n_units_out": 1,
                "seed": self._random_seed,
            },
            X_gt,
            X_syn,
        )


class DataLeakageXGB(AttackEvaluator):
    @staticmethod
    def name() -> str:
        return "data_leakage_xgb"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        return self._evaluate_leakage(
            XGBClassifier,
            {
                "n_jobs": -1,
                "eval_metric": "logloss",
            },
            XGBRegressor,
            {"n_jobs": -1},
            X_gt,
            X_syn,
        )


class DataLeakageLinear(AttackEvaluator):
    @staticmethod
    def name() -> str:
        return "data_leakage_linear"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        return self._evaluate_leakage(
            LogisticRegression,
            {"random_state": self._random_seed},
            LinearRegression,
            {},
            X_gt,
            X_syn,
        )
