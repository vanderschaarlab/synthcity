# third party
import numpy as np
import pandas as pd
import pytest
from generic_helpers import generate_fixtures
from sklearn.datasets import load_diabetes, load_iris

# synthcity absolute
from synthcity.metrics import PerformanceEvaluatorXGB
from synthcity.plugins import Plugin
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins.generic.plugin_pategan import plugin

plugin_name = "pategan"


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
    assert len(test_plugin.hyperparameter_space()) == 20


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_fit(test_plugin: Plugin) -> None:
    X = pd.DataFrame(load_iris()["data"])
    test_plugin.fit(GenericDataLoader(X))


def test_plugin_generate() -> None:
    test_plugin = plugin(generator_n_layers_hidden=1, generator_n_units_hidden=10)
    X = pd.DataFrame(load_diabetes()["data"])
    test_plugin.fit(GenericDataLoader(X))

    X_gen = test_plugin.generate()
    assert len(X_gen) == len(X)
    assert test_plugin.schema_includes(X_gen)

    X_gen = test_plugin.generate(50)
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)


def test_plugin_generate_constraints() -> None:
    test_plugin = plugin(generator_n_layers_hidden=1, generator_n_units_hidden=10)
    X = pd.DataFrame(load_iris()["data"])
    test_plugin.fit(GenericDataLoader(X))

    constraints = Constraints(
        rules=[
            ("0", "le", 6),
            ("0", "ge", 4.3),
            ("1", "le", 4.4),
            ("1", "ge", 3),
            ("2", "le", 5.5),
            ("2", "ge", 1.0),
            ("3", "le", 2),
            ("3", "ge", 0.1),
        ]
    )

    X_gen = test_plugin.generate(constraints=constraints).dataframe()
    assert len(X_gen) == len(X)
    assert test_plugin.schema_includes(X_gen)
    assert constraints.filter(X_gen).sum() == len(X_gen)

    X_gen = test_plugin.generate(count=50, constraints=constraints).dataframe()
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)
    assert constraints.filter(X_gen).sum() == len(X_gen)
    assert list(X_gen.columns) == list(X.columns)


def test_sample_hyperparams() -> None:
    for i in range(100):
        args = plugin.sample_hyperparameters()

        assert plugin(**args) is not None


@pytest.mark.slow
def test_eval_performance() -> None:
    results = []

    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    loader = GenericDataLoader(X)
    for retry in range(2):
        test_plugin = plugin()
        evaluator = PerformanceEvaluatorXGB()

        test_plugin.fit(loader)
        X_syn = test_plugin.generate()

        results.append(evaluator.evaluate(loader, X_syn)["syn_id"])

    assert np.mean(results) > 0.5
