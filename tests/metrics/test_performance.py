# stdlib
from typing import Type

# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris

# synthcity absolute
from synthcity.metrics.eval_performance import (
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

    assert "gt" in good_score
    assert "syn" in good_score

    assert good_score["gt"] > 0
    assert good_score["syn"] > 0

    sz = 100
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    score = evaluator.evaluate(
        X,
        X_rnd,
    )

    assert "gt" in score
    assert "syn" in score

    assert score["syn"] < good_score["syn"]

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

    assert "gt" in good_score
    assert "syn" in good_score

    sz = 100
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    score = evaluator.evaluate(
        X,
        X_rnd,
    )

    assert "gt" in score
    assert "syn" in score

    assert score["syn"] < good_score["syn"]
