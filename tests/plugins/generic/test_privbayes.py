# third party
import pytest
from generic_helpers import generate_fixtures, get_airfoil_dataset

# synthcity absolute
from synthcity.plugins import Plugin
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins.generic.plugin_privbayes import plugin

plugin_name = "privbayes"


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_sanity(test_plugin: Plugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_name(test_plugin: Plugin) -> None:
    assert test_plugin.name() == plugin_name


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_type(test_plugin: Plugin) -> None:
    assert test_plugin.type() == "generic"


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_hyperparams(test_plugin: Plugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 0


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_fit(test_plugin: Plugin) -> None:
    X = get_airfoil_dataset()
    test_plugin.fit(GenericDataLoader(X))


def test_plugin_generate_privbayes() -> None:
    X = get_airfoil_dataset()
    test_plugin = plugin()
    test_plugin.fit(GenericDataLoader(X))

    X_gen = test_plugin.generate(50)
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)
    assert sorted(list(X_gen.columns)) == sorted(list(X.columns))
