# stdlib
from typing import Type

# third party
import pytest
from sklearn.datasets import load_diabetes, load_iris

# synthcity absolute
from synthcity.metrics.privacy import (
    DeltaPresence,
    kAnonymization,
    kMap,
    lDiversity,
    select_outliers,
    select_quantiles,
)
from synthcity.plugins import Plugin, Plugins


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
@pytest.mark.parametrize(
    "method", ["isolation_forests", "local_outlier_factor", "elliptic_envelope"]
)
def test_select_outliers(test_plugin: Plugin, method: str) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    outlier_index = select_outliers(X, method=method)

    assert len(outlier_index.unique()) == 2
    assert outlier_index.sum() > 0


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_select_quantiles(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)

    original_duplicates = len(X.drop_duplicates())

    quantiles = select_quantiles(X)

    assert original_duplicates > len(quantiles.drop_duplicates())


@pytest.mark.parametrize(
    "evaluator_t",
    [
        DeltaPresence,
        kAnonymization,
        kMap,
        lDiversity,
    ],
)
@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluator(evaluator_t: Type, test_plugin: Plugin) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    test_plugin.fit(X)
    X_gen = test_plugin.generate(2 * len(X))

    evaluator = evaluator_t(sensitive_columns=["sex", "bmi"])

    score = evaluator.evaluate(X, X_gen)

    assert score > 0

    assert evaluator.type() == "privacy"
