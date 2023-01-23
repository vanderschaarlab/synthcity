# stdlib
import sys

# third party
import numpy as np
import pandas as pd
import pytest
from fhelpers import generate_fixtures
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics.eval import PerformanceEvaluatorXGB
from synthcity.plugins import Plugin
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins.privacy.plugin_dpgan import plugin
from synthcity.utils.serialization import load, save

plugin_name = "dpgan"
plugin_args = {
    "generator_n_layers_hidden": 1,
    "generator_n_units_hidden": 10,
    "n_iter": 10,
}


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
    assert len(test_plugin.hyperparameter_space()) == 14


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
def test_plugin_fit(test_plugin: Plugin) -> None:
    X = pd.DataFrame(load_iris()["data"])
    test_plugin.fit(GenericDataLoader(X))


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
@pytest.mark.parametrize("serialize", [True, False])
def test_plugin_generate(test_plugin: Plugin, serialize: bool) -> None:
    X = pd.DataFrame(load_iris()["data"])
    test_plugin.fit(GenericDataLoader(X))

    if serialize:
        saved = save(test_plugin)
        test_plugin = load(saved)

    X_gen = test_plugin.generate()
    assert len(X_gen) == len(X)
    assert test_plugin.schema_includes(X_gen)

    X_gen = test_plugin.generate(50)
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
def test_plugin_generate_constraints(test_plugin: Plugin) -> None:
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
@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
def test_eval_performance_dpgan() -> None:
    results = []

    Xraw, y = load_iris(return_X_y=True, as_frame=True)
    Xraw["target"] = y
    X = GenericDataLoader(Xraw)

    for retry in range(2):
        test_plugin = plugin(n_iter=300)
        evaluator = PerformanceEvaluatorXGB()

        test_plugin.fit(X)
        X_syn = test_plugin.generate()

        results.append(evaluator.evaluate(X, X_syn)["syn_id"])

    print(results)
    assert np.mean(results) > 0.5


@pytest.mark.slow
def test_plugin_conditional_dpgan() -> None:
    test_plugin = plugin(generator_n_units_hidden=5)
    Xraw, y = load_iris(as_frame=True, return_X_y=True)
    Xraw["target"] = y

    X = GenericDataLoader(Xraw)
    test_plugin.fit(X, cond=y)

    X_gen = test_plugin.generate(2 * len(X))
    assert len(X_gen) == 2 * len(X)
    assert test_plugin.schema_includes(X_gen)

    count = 10
    X_gen = test_plugin.generate(count, cond=np.ones(count))
    assert len(X_gen) == count

    assert (X_gen["target"] == 1).sum() >= 0.8 * count
