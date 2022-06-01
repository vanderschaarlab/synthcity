# stdlib
from typing import Type

# third party
import pytest
from sklearn.datasets import load_diabetes

# synthcity absolute
from synthcity.metrics.eval_privacy import (
    DeltaPresence,
    IdentifiabilityScore,
    kAnonymization,
    kMap,
    lDiversityDistinct,
)
from synthcity.plugins import Plugin, Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader


@pytest.mark.parametrize(
    "evaluator_t",
    [
        DeltaPresence,
        kAnonymization,
        kMap,
        lDiversityDistinct,
        IdentifiabilityScore,
    ],
)
@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluator(evaluator_t: Type, test_plugin: Plugin) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    Xloader = GenericDataLoader(X, sensitive_features=["sex", "bmi"])
    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(2 * len(X))

    evaluator = evaluator_t()

    score = evaluator.evaluate(Xloader, X_gen)

    for submetric in score:
        assert score[submetric] > 0

    assert evaluator.type() == "privacy"
