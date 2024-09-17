# stdlib
import platform
from typing import Any, Dict

# third party
import numpy as np
from pydantic import validate_arguments
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import r2_score
from xgboost import XGBClassifier, XGBRegressor
import pandas as pd

# synthcity absolute
from synthcity.metrics._utils import evaluate_auc
from synthcity.metrics.core import MetricEvaluator
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.models.mlp import MLP
from synthcity.utils.serialization import load_from_file, save_to_file
from synthcity.plugins.core.models.tabular_encoder import preprocess_prediction


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

            # if column is discrete do classification
            if col in X_gt.discrete_features:
                task_type = "classification"
                # dont try to one hot target column in input features
                discrete = [x for x in X_gt.discrete_features if x != col]
            else:
                task_type = "regression"
                discrete = X_gt.discrete_features.copy()

            # preprocess input data
            target = X_syn[col]
            keys_data = X_syn.drop(columns=[col])
            test_target = X_gt[col]
            test_keys_data = X_gt.drop(columns=[col])
            keys_data, test_keys_data = preprocess_prediction(
                train=keys_data.dataframe(),
                test=test_keys_data.dataframe(),
                discrete_features=discrete,
            )

            # setup target and model for classification / regression
            if task_type == "classification":

                # if some labels occur in test data which do not appear in train data, remove those datapoints
                test_keys_data = test_keys_data[test_target.isin(target.unique())]
                test_target = test_target[test_target.isin(target.unique())]

                encoder = LabelEncoder()
                target = encoder.fit_transform(target)
                test_target = encoder.transform(test_target)
                if "n_units_out" in classifier_args:
                    # TBD: fix MLP
                    classifier_args["n_units_out"] = len(np.unique(target))
                    classifier_args["n_units_in"] = keys_data.shape[1]
                model = classifier_template(**classifier_args)
            else:
                encoder = MinMaxScaler(feature_range=(-1, 1))
                target = pd.Series(
                    encoder.fit_transform(target.to_frame()).flatten(),
                    index=target.index,
                )
                test_target = pd.Series(
                    encoder.fit_transform(test_target.to_frame()).flatten(),
                    index=test_target.index,
                )
                model = regressor_template(**regressor_args)
            model.fit(keys_data.values, np.asarray(target))
            # get predictions and scores
            if task_type == "classification":
                preds = model.predict_proba(test_keys_data.values)
                output_, _ = evaluate_auc(
                    np.asarray(test_target),
                    np.asarray(preds),
                    classes=sorted(set(np.asarray(target))),
                )
            else:
                preds = model.predict(test_keys_data.values)
                output_ = r2_score(np.asarray(test_target), np.asarray(preds))

            output.append(output_)

        if len(output) == 0:
            return {}

        # save results per feature
        results = {}
        for num, col in enumerate(X_gt.sensitive_features):
            self.col = col
            results[self.col] = output[num]

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
