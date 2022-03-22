# stdlib
from typing import Callable

# third party
import pytest
from sklearn.datasets import load_diabetes

# synthcity absolute
from synthcity.metrics.attacks import (
    evaluate_sensitive_data_leakage_linear,
    evaluate_sensitive_data_leakage_mlp,
    evaluate_sensitive_data_leakage_xgb,
)
from synthcity.plugins import Plugins


@pytest.mark.parametrize(
    "evaluator",
    [
        evaluate_sensitive_data_leakage_linear,
        evaluate_sensitive_data_leakage_xgb,
        evaluate_sensitive_data_leakage_mlp,
    ],
)
def test_evaluate_sensitive_data_leakage(evaluator: Callable) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    # Sampler
    test_plugin = Plugins().get("dummy_sampler")
    test_plugin.fit(X)
    X_gen = test_plugin.generate(2 * len(X))

    score = evaluator(
        X.drop(columns=["target"]),
        X_gen.drop(columns=["target"]),
    )
    assert score == 0

    score = evaluator(
        X.drop(columns=["target"]),
        X_gen.drop(columns=["target"]),
        sensitive_columns=["sex"],
    )
    assert score > 0.5

    score = evaluator(
        X.drop(columns=["target"]),
        X_gen.drop(columns=["target"]),
        sensitive_columns=["age"],
    )
    assert score < 1

    # Random noise

    test_plugin = Plugins().get("random_noise")
    test_plugin.fit(X)
    X_gen = test_plugin.generate(2 * len(X))

    score = evaluator(
        X.drop(columns=["target"]),
        X_gen.drop(columns=["target"]),
        sensitive_columns=["sex"],
    )
    assert score < 1
