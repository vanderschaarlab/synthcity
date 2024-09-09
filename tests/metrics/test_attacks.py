# stdlib
from typing import Type

# third party
import pytest
from sklearn.datasets import load_diabetes

# synthcity absolute
from synthcity.metrics.eval_attacks import (
    DataLeakageLinear,
    DataLeakageMLP,
    DataLeakageXGB,
)
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader


@pytest.mark.parametrize("reduction", ["mean", "max", "min"])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        DataLeakageLinear,
        DataLeakageXGB,
        DataLeakageMLP,
    ],
)
def test_reduction(reduction: str, evaluator_t: Type) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    # Sampler
    test_plugin = Plugins().get("dummy_sampler")
    Xloader = GenericDataLoader(X, sensitive_features=["sex"])

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(10)

    evaluator = evaluator_t(
        reduction=reduction,
        use_cache=False,
    )

    score = evaluator.evaluate(
        Xloader,
        X_gen,
    )

    assert reduction in score

    def_score = evaluator.evaluate_default(Xloader, X_gen)

    assert def_score == score[reduction]


@pytest.mark.parametrize(
    "evaluator_t",
    [
        DataLeakageLinear,
        DataLeakageXGB,
        DataLeakageMLP,
    ],
)
def test_evaluate_sensitive_data_leakage(evaluator_t: Type) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    # Sampler
    test_plugin = Plugins().get("dummy_sampler")
    Xloader = GenericDataLoader(X, sensitive_features=["sex"])

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(2 * len(X))

    evaluator = evaluator_t()

    score = evaluator.evaluate(
        Xloader,
        X_gen,
    )["mean"]
    assert score > 0.5
    assert score < 1

    # Random noise
    test_plugin = Plugins().get("uniform_sampler")
    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(2 * len(X))

    score = evaluator.evaluate(
        Xloader,
        X_gen,
    )["mean"]
    assert score < 1

    assert evaluator.type() == "attack"
    assert evaluator.direction() == "minimize"
