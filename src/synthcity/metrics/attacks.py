# stdlib
from typing import Any, Dict, List

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

# synthcity absolute
from synthcity.plugins.models.mlp import MLP


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_sensitive_data_leakage(
    classifier_template: Any,
    classifier_args: Dict,
    regressor_template: Any,
    regressor_args: Dict,
    X_gt: pd.DataFrame,
    X_syn: pd.DataFrame,
    sensitive_columns: List[str] = [],
) -> float:
    if sensitive_columns == []:
        return 0

    output = []
    for col in sensitive_columns:
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

    return np.mean(output)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_sensitive_data_leakage_mlp(
    X_gt: pd.DataFrame,
    X_syn: pd.DataFrame,
    sensitive_columns: List[str] = [],
) -> float:
    return evaluate_sensitive_data_leakage(
        MLP,
        {
            "task_type": "classification",
            "n_units_in": X_gt.shape[1] - 1,
            "n_units_out": 0,
        },
        MLP,
        {"task_type": "regression", "n_units_in": X_gt.shape[1] - 1, "n_units_out": 1},
        X_gt,
        X_syn,
        sensitive_columns,
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_sensitive_data_leakage_xgb(
    X_gt: pd.DataFrame,
    X_syn: pd.DataFrame,
    sensitive_columns: List[str] = [],
) -> float:
    return evaluate_sensitive_data_leakage(
        XGBClassifier,
        {"n_jobs": 1},
        XGBRegressor,
        {"n_jobs": 1},
        X_gt,
        X_syn,
        sensitive_columns,
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_sensitive_data_leakage_linear(
    X_gt: pd.DataFrame,
    X_syn: pd.DataFrame,
    sensitive_columns: List[str] = [],
) -> float:
    return evaluate_sensitive_data_leakage(
        LogisticRegression,
        {},
        LinearRegression,
        {},
        X_gt,
        X_syn,
        sensitive_columns,
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_membership_inference_attack(
    X_gt: pd.DataFrame,
    X_syn: pd.DataFrame,
    sensitive_columns: List[str] = [],
) -> float:
    # TODO
    pass
