# third party
import pandas as pd
import pytest
from fhelpers import generate_fixtures
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.plugins import Plugin
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins.fairness.plugin_decaf import plugin

plugin_name = "decaf"
plugin_args = {"n_iter": 50}


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_sanity(test_plugin: Plugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_name(test_plugin: Plugin) -> None:
    assert test_plugin.name() == plugin_name


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_type(test_plugin: Plugin) -> None:
    assert test_plugin.type() == "fairness"


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_hyperparams(test_plugin: Plugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 2


@pytest.mark.parametrize(
    "struct_learning_search_method",
    ["hillclimb", "d-struct"],
)
@pytest.mark.parametrize("struct_learning_score", ["k2", "bdeu"])
@pytest.mark.slow
def test_plugin_fit(
    struct_learning_search_method: str, struct_learning_score: str
) -> None:
    test_plugin = plugin(
        n_iter=50,
        struct_learning_search_method=struct_learning_search_method,
        struct_learning_score=struct_learning_score,
        struct_learning_enabled=True,
    )
    X = pd.DataFrame(load_iris()["data"])
    test_plugin.fit(GenericDataLoader(X))


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
def test_plugin_generate(test_plugin: Plugin) -> None:
    X = pd.DataFrame(load_iris()["data"])
    test_plugin.fit(GenericDataLoader(X))

    X_gen = test_plugin.generate()
    assert len(X_gen) == len(X)
    assert test_plugin.schema_includes(X_gen)

    X_gen = test_plugin.generate(50)
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)
    assert list(X_gen.columns) == list(X.columns)


@pytest.mark.parametrize(
    "struct_learning_search_method",
    ["hillclimb", "d-struct"],
)
def test_get_dag(struct_learning_search_method: str) -> None:
    test_plugin = plugin(
        struct_learning_enabled=True,
        struct_learning_search_method=struct_learning_search_method,
        **plugin_args
    )

    X = pd.DataFrame(load_iris()["data"])
    dag = test_plugin._get_dag(X)

    print(dag)


@pytest.mark.parametrize(
    "struct_learning_search_method",
    ["hillclimb", "d-struct"],
)
def test_plugin_generate_and_learn_dag(struct_learning_search_method: str) -> None:
    test_plugin = plugin(
        struct_learning_enabled=True,
        struct_learning_search_method=struct_learning_search_method,
        **plugin_args
    )

    X = pd.DataFrame(load_iris()["data"])
    test_plugin.fit(GenericDataLoader(X))

    X_gen = test_plugin.generate()
    assert len(X_gen) == len(X)
    assert test_plugin.schema_includes(X_gen)

    X_gen = test_plugin.generate(50)
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)
    assert list(X_gen.columns) == list(X.columns)
