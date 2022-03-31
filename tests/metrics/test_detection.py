# stdlib
from typing import Type

# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics.eval_detection import (
    SyntheticDetectionGMM,
    SyntheticDetectionMLP,
    SyntheticDetectionXGB,
)
from synthcity.plugins import Plugin, Plugins


@pytest.mark.parametrize("reduction", ["mean", "max", "min"])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        SyntheticDetectionXGB,
    ],
)
def test_detect_reduction(reduction: str, evaluator_t: Type) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin = Plugins().get("marginal_distributions")
    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    evaluator = evaluator_t(reduction=reduction)

    score = evaluator.evaluate(
        X,
        X_gen,
    )

    assert reduction in score


@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        SyntheticDetectionXGB,
        SyntheticDetectionGMM,
        SyntheticDetectionMLP,
    ],
)
def test_detect_synth(test_plugin: Plugin, evaluator_t: Type) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    evaluator = evaluator_t()

    good_score = evaluator.evaluate(
        X,
        X_gen,
    )["mean"]

    assert good_score > 0
    assert good_score <= 1

    sz = 100
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    score = evaluator.evaluate(
        X,
        X_rnd,
    )["mean"]

    assert score > 0
    assert score <= 1
    assert good_score < score

    assert evaluator.type() == "detection"
    assert evaluator.direction() == "minimize"
