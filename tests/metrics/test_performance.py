# stdlib
from typing import Optional, Type

# third party
import numpy as np
import pandas as pd
import pytest
from lifelines.datasets import load_rossi
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
    print(X.columns)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    evaluator = evaluator_t()
    good_score = evaluator.evaluate(
        X,
        X,
        X_gen,
    )

    assert "gt" in good_score
    assert "syn_id" in good_score
    assert "syn_ood" in good_score

    assert good_score["gt"] > 0
    assert good_score["syn_id"] > 0
    assert good_score["syn_ood"] > 0

    sz = 100
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    score = evaluator.evaluate(
        X,
        X,
        X_rnd,
    )

    assert "gt" in score
    assert "syn_id" in score
    assert "syn_ood" in score

    assert score["syn_id"] < good_score["syn_id"]
    assert score["syn_ood"] < good_score["syn_ood"]

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
        X,
        X_gen,
    )

    assert "gt" in good_score
    assert "syn_id" in good_score
    assert "syn_ood" in good_score

    sz = 1000
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    score = evaluator.evaluate(
        X,
        X,
        X_rnd,
    )
    print(good_score, score)

    assert "gt" in score
    assert "syn_id" in score
    assert "syn_ood" in score

    assert score["syn_id"] <= good_score["syn_id"]
    assert score["syn_ood"] <= good_score["syn_ood"]


@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        PerformanceEvaluatorLinear,
        PerformanceEvaluatorMLP,
        PerformanceEvaluatorXGB,
    ],
)
def test_evaluate_performance_survival_analysis(
    test_plugin: Plugin, evaluator_t: Type
) -> None:
    X = load_rossi()
    T = X["week"]

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)
    time_horizons = np.linspace(T.min(), T.max(), num=4)[1:-1].tolist()

    evaluator = evaluator_t(
        task_type="survival_analysis",
        target_column="arrest",
        time_to_event_column="week",
        time_horizons=time_horizons,
    )
    good_score = evaluator.evaluate(
        X,
        X,
        X_gen,
    )

    assert "gt" in good_score
    assert "syn_id" in good_score
    assert "syn_ood" in good_score

    sz = 100
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    X_rnd["arrest"] = 1
    score = evaluator.evaluate(
        X,
        X,
        X_rnd,
    )

    assert "gt" in score
    assert "syn_id" in score
    assert "syn_ood" in score

    assert score["syn_id"] < 1
    assert score["syn_ood"] < 1
    assert good_score["gt"] < 1


@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        PerformanceEvaluatorLinear,
        PerformanceEvaluatorXGB,
    ],
)
@pytest.mark.parametrize("target", [None, "target", "sepal width (cm)"])
def test_evaluate_performance_custom_labels(
    test_plugin: Plugin, evaluator_t: Type, target: Optional[str]
) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    evaluator = evaluator_t(target_column=target)

    assert evaluator._target_column == target

    good_score = evaluator.evaluate(
        X,
        X,
        X_gen,
    )

    assert "gt" in good_score
    assert "syn_id" in good_score
    assert "syn_ood" in good_score

    # Test fail

    evaluator = evaluator_t(target_column="invalid_col")
    with pytest.raises(ValueError):
        evaluator.evaluate(
            X,
            X,
            X_gen,
        )
