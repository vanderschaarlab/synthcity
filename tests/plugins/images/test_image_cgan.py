# third party
import numpy as np
import pytest
from img_helpers import generate_fixtures
from torchvision import datasets

# synthcity absolute
from synthcity.plugins import Plugin
from synthcity.plugins.core.dataloader import ImageDataLoader
from synthcity.plugins.images.plugin_image_cgan import plugin

plugin_name = "image_cgan"

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


@pytest.mark.parametrize("height", [32, 64, 128])
@pytest.mark.slow
def test_plugin_fit(height: int) -> None:
    test_plugin = plugin(n_iter=5)

    X = ImageDataLoader(dataset, height=height).sample(100)

    test_plugin.fit(X)


def test_plugin_generate() -> None:
    test_plugin = plugin(n_iter=10, n_units_latent=13)

    X = ImageDataLoader(dataset).sample(100)

    test_plugin.fit(X)

    X_gen = test_plugin.generate()
    assert len(X_gen) == len(X)
    assert X_gen.shape == X.shape

    X_gen = test_plugin.generate(50)
    assert len(X_gen) == 50


def test_plugin_generate_with_conditional() -> None:
    test_plugin = plugin(n_iter=10, n_units_latent=13)

    X = ImageDataLoader(dataset).sample(100)
    cond = X.unpack().labels()

    test_plugin.fit(X, cond=cond)

    cnt = 50
    X_gen = test_plugin.generate(cnt, cond=np.ones(cnt))
    assert len(X_gen) == 50


@pytest.mark.slow
def test_plugin_generate_with_stop_conditional() -> None:
    test_plugin = plugin(n_iter=10, n_units_latent=13, n_iter_print=2)

    X = ImageDataLoader(dataset).sample(100)
    cond = X.unpack().labels()

    test_plugin.fit(X, cond=cond)

    cnt = 50
    X_gen = test_plugin.generate(cnt, cond=np.ones(cnt))
    assert len(X_gen) == 50


def test_sample_hyperparams() -> None:
    for i in range(100):
        args = plugin.sample_hyperparameters()

        assert plugin(**args) is not None
