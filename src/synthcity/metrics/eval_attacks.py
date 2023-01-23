# stdlib
import platform
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
from synthcity.utils.serialization import load_from_file, save_to_file


class AttackEvaluator(MetricEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_attacks.AttackEvaluator
        :parts: 1

    Evaluating the risk of attribute inference attack.

    This class evaluates the risk of a type of privacy attack, known as attribute inference attack.
    In this setting, the attacker has access to the synthetic dataset as well as partial information about the real data
    (quasi-identifiers). The attacker seeks to uncover the sensitive attributes of the real data using these two pieces
    of information.
    """

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
        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{X_gt.hash()}_{X_syn.hash()}_{self._reduction}_{platform.python_version()}.bkp"
        )
        if self.use_cache(cache_file):
            return load_from_file(cache_file)

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

        results = {self._reduction: self.reduction()(output)}

        save_to_file(cache_file, results)

        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_default(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> float:
        return self.evaluate(X_gt, X_syn)[self._reduction]


class DataLeakageMLP(AttackEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_attacks.DataLeakageMLP
        :parts: 1

    Data leakage test using a neural net.
    """

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
                "random_state": self._random_state,
            },
            MLP,
            {
                "task_type": "regression",
                "n_units_in": X_gt.shape[1] - 1,
                "n_units_out": 1,
                "random_state": self._random_state,
            },
            X_gt,
            X_syn,
        )


class DataLeakageXGB(AttackEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_attacks.DataLeakageXGB
        :parts: 1

    Data leakage test using XGBoost
    """

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
    """
    .. inheritance-diagram:: synthcity.metrics.eval_attacks.DataLeakageLinear
        :parts: 1


    Data leakage test using a linear model
    """

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
            {"random_state": self._random_state},
            LinearRegression,
            {},
            X_gt,
            X_syn,
        )
