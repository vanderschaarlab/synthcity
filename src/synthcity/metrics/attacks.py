# stdlib
from typing import List

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.preprocessing import LabelEncoder

# synthcity absolute
from synthcity.plugins.models.mlp import MLP


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_sensitive_data_leakage(
    X_gt: pd.DataFrame,
    y_gt: pd.Series,
    X_syn: pd.DataFrame,
    y_syn: pd.Series,
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
        else:
            task_type = "regression"

        model = MLP(task_type=task_type)
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
def evaluate_generalized_cap(
    X_gt: pd.DataFrame,
    y_gt: pd.Series,
    X_syn: pd.DataFrame,
    y_syn: pd.Series,
) -> float:
    pass
