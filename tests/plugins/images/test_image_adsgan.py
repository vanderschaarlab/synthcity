# stdlib
import os
import sys
from pathlib import Path

# third party
import numpy as np
import pytest
from img_helpers import generate_fixtures
from torchvision import datasets

# synthcity absolute
from synthcity.plugins import Plugin
from synthcity.plugins.core.dataloader import ImageDataLoader
from synthcity.plugins.images.plugin_image_adsgan import plugin

plugin_name = "image_adsgan"


def get_mnist() -> datasets.MNIST:
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
    return dataset


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


@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
def test_plugin_fit() -> None:
    dataset = get_mnist()
    test_plugin = plugin(n_iter=5)

    X = ImageDataLoader(dataset).sample(100)

    test_plugin.fit(X)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
def test_plugin_generate() -> None:
    dataset = get_mnist()
    test_plugin = plugin(n_iter=10, n_units_latent=13)

    X = ImageDataLoader(dataset).sample(100)

    test_plugin.fit(X)

    X_gen = test_plugin.generate()
    assert len(X_gen) == len(X)
    assert X_gen.shape == X.shape

    X_gen = test_plugin.generate(50)
    assert len(X_gen) == 50


@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
@pytest.mark.slow_2
@pytest.mark.slow
def test_plugin_generate_with_conditional() -> None:
    dataset = get_mnist()
    test_plugin = plugin(n_iter=10, n_units_latent=13)

    X = ImageDataLoader(dataset).sample(100)
    cond = X.unpack().labels()

    test_plugin.fit(X, cond=cond)

    cnt = 50
    X_gen = test_plugin.generate(cnt, cond=np.ones(cnt))
    assert len(X_gen) == 50


@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
@pytest.mark.slow_2
@pytest.mark.slow
def test_plugin_generate_with_stop_conditional() -> None:
    dataset = get_mnist()
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
