# stdlib
import sys

# third party
import numpy as np
import pandas as pd
import pytest
from fhelpers import generate_fixtures, get_airfoil_dataset
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics.eval import PerformanceEvaluatorXGB
from synthcity.plugins import Plugin
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins.privacy.plugin_adsgan import plugin

plugin_name = "adsgan"


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
    assert len(test_plugin.hyperparameter_space()) == 11


def test_plugin_fit() -> None:
    test_plugin = plugin(
        n_iter=100, generator_n_layers_hidden=1, generator_n_units_hidden=10
    )

    df = pd.DataFrame(load_iris()["data"])
    X = GenericDataLoader(df)

    test_plugin.fit(X)


def test_plugin_generate() -> None:
    test_plugin = plugin(
        n_iter=100, generator_n_layers_hidden=1, generator_n_units_hidden=10
    )

    df = get_airfoil_dataset()
    X = GenericDataLoader(df)

    test_plugin.fit(X)

    X_gen = test_plugin.generate()
    assert len(X_gen) == len(X)
    assert X_gen.shape[1] == df.shape[1]
    assert test_plugin.schema_includes(X_gen)

    X_gen = test_plugin.generate(50)
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
def test_plugin_conditional_adsgan() -> None:
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


def test_plugin_generate_constraints() -> None:
    test_plugin = plugin(
        n_iter=100, generator_n_layers_hidden=1, generator_n_units_hidden=10
    )

    Xraw = pd.DataFrame(load_iris()["data"])
    X = GenericDataLoader(Xraw)
    test_plugin.fit(X)

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

    X_gen = test_plugin.generate(constraints=constraints)
    assert len(X_gen) == len(X)
    assert test_plugin.schema_includes(X_gen)
    assert constraints.filter(X_gen.dataframe()).sum() == len(X_gen)

    X_gen = test_plugin.generate(count=50, constraints=constraints)
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)
    assert constraints.filter(X_gen.dataframe()).sum() == len(X_gen)
    assert list(X_gen.columns) == list(X.columns)


def test_sample_hyperparams() -> None:
    for i in range(100):
        args = plugin.sample_hyperparameters()

        assert plugin(**args) is not None


@pytest.mark.slow
@pytest.mark.parametrize("compress_dataset", [True, False])
def test_eval_performance(compress_dataset: bool) -> None:
    results = []

    Xraw, y = load_iris(return_X_y=True, as_frame=True)
    Xraw["target"] = y
    X = GenericDataLoader(Xraw)

    for retry in range(2):
        test_plugin = plugin(n_iter=5000, compress_dataset=compress_dataset)
        evaluator = PerformanceEvaluatorXGB()

        test_plugin.fit(X)
        X_syn = test_plugin.generate()

        results.append(evaluator.evaluate(X, X_syn)["syn_id"])

    print(plugin.name(), results)
    assert np.mean(results) > 0.8
