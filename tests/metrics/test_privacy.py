# stdlib
import os
import sys
from pathlib import Path
from typing import Type

# third party
import numpy as np
import pytest
from sklearn.datasets import load_iris
from torchvision import datasets

# synthcity absolute
from synthcity.metrics.eval_privacy import (
    DeltaPresence,
    DomiasMIABNAF,
    DomiasMIAKDE,
    DomiasMIAPrior,
    IdentifiabilityScore,
    kAnonymization,
    kMap,
    lDiversityDistinct,
)
from synthcity.plugins import Plugin, Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader, ImageDataLoader


@pytest.mark.skipif(sys.platform != "linux", reason="Only test on linux for speed")
@pytest.mark.parametrize(
    "evaluator_t",
    [
        DeltaPresence,
        kAnonymization,
        kMap,
        lDiversityDistinct,
        IdentifiabilityScore,
        DomiasMIABNAF,
        DomiasMIAKDE,
        DomiasMIAPrior,
    ],
)
@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluator(evaluator_t: Type, test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)

    Xloader = GenericDataLoader(
        X, sensitive_features=["sepal length (cm)", "sepal width (cm)"]
    )
    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(2 * len(X))

    evaluator = evaluator_t(
        use_cache=False,
    )
    if "DomiasMIA" in evaluator.name():
        X_ref_syn = test_plugin.generate(2 * len(X))
        score = evaluator.evaluate(
            Xloader,
            X_gen,
            Xloader.train(),
            X_ref_syn,
            reference_size=10,
        )
    else:
        score = evaluator.evaluate(Xloader, X_gen)

    for submetric in score:
        assert score[submetric] > 0

    assert evaluator.type() == "privacy"

    if "DomiasMIA" in evaluator.name():
        X_ref_syn = test_plugin.generate(2 * len(X))
        def_score = evaluator.evaluate_default(
            Xloader,
            X_gen,
            Xloader.train(),
            X_ref_syn,
            reference_size=10,
        )
    else:
        def_score = evaluator.evaluate_default(Xloader, X_gen)

    assert isinstance(def_score, (float, int))


@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
def test_image_support() -> None:
    # Get the MNIST dataset directory from an environment variable
    mnist_dir = os.getenv(
        "MNIST_DATA_DIR", "."
    )  # Default to current directory if not set

    # Check if the MNIST dataset is already downloaded
    mnist_path = Path(mnist_dir) / "MNIST" / "processed"
    if not mnist_path.exists():
        dataset = datasets.MNIST(mnist_dir, download=True)
    else:
        dataset = datasets.MNIST(mnist_dir, train=True)

    X1 = ImageDataLoader(dataset).sample(100)
    X2 = ImageDataLoader(dataset).sample(100)

    for evaluator in [
        IdentifiabilityScore,
    ]:
        score = evaluator().evaluate(X1, X2)
        assert isinstance(score, dict)
        for k in score:
            assert score[k] >= 0
            assert not np.isnan(score[k])
