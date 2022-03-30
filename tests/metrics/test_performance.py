# stdlib
from typing import Type

# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris

# synthcity absolute
from synthcity.metrics.performance import (
    PerformanceEvaluatorLinear,
    PerformanceEvaluatorMLP,
    PerformanceEvaluatorXGB,
)
from synthcity.plugins import Plugin, Plugins


@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        PerformanceEvaluatorLinear,
        PerformanceEvaluatorMLP,
        PerformanceEvaluatorXGB,
    ],
)
def test_evaluate_performance_classifier(
    test_plugin: Plugin, evaluator_t: Type
) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    evaluator = evaluator_t()
    good_score = evaluator.evaluate(
        X,
        X_gen,
    )

    assert np.abs(good_score) < 1

    sz = 100
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    score = evaluator.evaluate(
        X,
        X_rnd,
    )

    assert np.abs(good_score) < 1
    assert score > good_score

    assert evaluator.type() == "performance"
    assert evaluator.direction() == "maximize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        PerformanceEvaluatorLinear,
        PerformanceEvaluatorMLP,
        PerformanceEvaluatorXGB,
    ],
)
def test_evaluate_performance_regression(
    test_plugin: Plugin, evaluator_t: Type
) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    evaluator = evaluator_t()
    good_score = evaluator.evaluate(
        X,
        X_gen,
    )

    assert np.abs(good_score) < 1

    sz = 100
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    score = evaluator.evaluate(
        X,
        X_rnd,
    )

    assert np.abs(good_score) < 1
    assert score > good_score
