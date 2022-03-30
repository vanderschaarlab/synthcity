# stdlib
from typing import Type

# third party
import pytest
from sklearn.datasets import load_diabetes

# synthcity absolute
from synthcity.metrics.attacks import DataLeakageLinear, DataLeakageMLP, DataLeakageXGB
from synthcity.plugins import Plugins


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
    test_plugin.fit(X)
    X_gen = test_plugin.generate(2 * len(X))

    evaluator = evaluator_t(sensitive_columns=["sex"])

    score = evaluator.evaluate(
        X,
        X_gen,
    )
    assert score > 0.5
    assert score < 1

    # Random noise

    test_plugin = Plugins().get("uniform_sampler")
    test_plugin.fit(X)
    X_gen = test_plugin.generate(2 * len(X))

    score = evaluator.evaluate(
        X,
        X_gen,
    )
    assert score < 1

    assert evaluator.type() == "attack"
    assert evaluator.direction() == "minimize"
