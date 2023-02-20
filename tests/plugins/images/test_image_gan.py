# third party
import numpy as np
import pytest
from img_helpers import generate_fixtures
from torchvision import datasets

# synthcity absolute
from synthcity.metrics.eval import PerformanceEvaluatorXGB
from synthcity.plugins import Plugin
from synthcity.plugins.core.dataloader import ImageDataLoader
from synthcity.plugins.images.plugin_image_gan import plugin

plugin_name = "image_gan"

dataset = datasets.MNIST(".", download=True)


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_sanity(test_plugin: Plugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_name(test_plugin: Plugin) -> None:
    assert test_plugin.name() == plugin_name


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_type(test_plugin: Plugin) -> None:
    assert test_plugin.type() == "images"


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_hyperparams(test_plugin: Plugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 6


def test_plugin_fit() -> None:
    test_plugin = plugin(n_iter=5)

    X = ImageDataLoader(dataset).sample(100)

    test_plugin.fit(X)


def test_plugin_generate() -> None:
    test_plugin = plugin(n_iter=10)

    X = ImageDataLoader(dataset).sample(100)

    test_plugin.fit(X)

    X_gen = test_plugin.generate()
    assert len(X_gen) == len(X)
    assert X_gen.shape == X.shape

    X_gen = test_plugin.generate(50)
    assert len(X_gen) == 50


def test_sample_hyperparams() -> None:
    for i in range(100):
        args = plugin.sample_hyperparameters()

        assert plugin(**args) is not None


@pytest.mark.slow
def test_eval_performance() -> None:
    results = []

    X = ImageDataLoader(dataset)

    for retry in range(2):
        test_plugin = plugin(n_iter=500)
        evaluator = PerformanceEvaluatorXGB()

        test_plugin.fit(X)
        X_syn = test_plugin.generate()

        results.append(evaluator.evaluate(X, X_syn)["syn_id"])

    print(plugin.name(), results)
    assert np.mean(results) > 0.8
