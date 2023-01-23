# third party
import pytest
from fhelpers import generate_fixtures
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.plugins import Plugin
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins.privacy.plugin_privbayes import plugin

plugin_name = "privbayes"


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_sanity(test_plugin: Plugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_name(test_plugin: Plugin) -> None:
    assert test_plugin.name() == plugin_name


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_type(test_plugin: Plugin) -> None:
    assert test_plugin.type() == "privacy"


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_hyperparams(test_plugin: Plugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 0


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_fit(test_plugin: Plugin) -> None:
    X, _ = load_iris(as_frame=True, return_X_y=True)
    X["sepal length (cm)"] = X["sepal length (cm)"].astype(str)  # test categoricals

    test_plugin.fit(GenericDataLoader(X))


def test_plugin_generate_privbayes() -> None:
    X, _ = load_iris(as_frame=True, return_X_y=True)
    test_plugin = plugin(K=2)
    test_plugin.fit(GenericDataLoader(X))

    X_gen = test_plugin.generate(50)
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)
    assert sorted(list(X_gen.columns)) == sorted(list(X.columns))
