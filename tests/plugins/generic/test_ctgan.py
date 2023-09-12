# stdlib
import random
import sys
from datetime import datetime, timedelta

# third party
import numpy as np
import pandas as pd
import pytest
from generic_helpers import generate_fixtures, get_airfoil_dataset
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics.eval import PerformanceEvaluatorXGB
from synthcity.plugins import Plugin
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins.generic.plugin_ctgan import plugin
from synthcity.utils.serialization import load, save

plugin_name = "ctgan"
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
    assert test_plugin.type() == "generic"


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_hyperparams(test_plugin: Plugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 14


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
def test_plugin_fit(test_plugin: Plugin) -> None:
    X = get_airfoil_dataset()
    test_plugin.fit(GenericDataLoader(X))


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
@pytest.mark.parametrize("serialize", [True, False])
def test_plugin_generate(test_plugin: Plugin, serialize: bool) -> None:
    X = get_airfoil_dataset()
    test_plugin.fit(GenericDataLoader(X))

    if serialize:
        saved = save(test_plugin)
        test_plugin = load(saved)

    X_gen = test_plugin.generate()
    assert len(X_gen) == len(X)
    assert X_gen.shape[1] == X.shape[1]
    assert test_plugin.schema_includes(X_gen)

    X_gen = test_plugin.generate(50)
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)

    # generate with random seed
    X_gen1 = test_plugin.generate(50, random_state=0)
    X_gen2 = test_plugin.generate(50, random_state=0)
    X_gen3 = test_plugin.generate(50)
    assert (X_gen1.numpy() == X_gen2.numpy()).all()
    assert (X_gen1.numpy() != X_gen3.numpy()).any()


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
def test_plugin_generate_constraints_ctgan(test_plugin: Plugin) -> None:
    X, y = load_iris(as_frame=True, return_X_y=True)
    X["target"] = y
    test_plugin.fit(GenericDataLoader(X))

    constraints = Constraints(
        rules=[
            ("target", "eq", 1),
        ]
    )

    X_gen = test_plugin.generate(constraints=constraints).dataframe()
    assert len(X_gen) == len(X)
    assert test_plugin.schema_includes(X_gen)
    assert constraints.filter(X_gen).sum() == len(X_gen)
    assert (X_gen["target"] == 1).all()

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
@pytest.mark.parametrize("compress_dataset", [True, False])
def test_eval_performance_ctgan(compress_dataset: bool) -> None:
    results = []

    Xraw, y = load_iris(return_X_y=True, as_frame=True)
    Xraw["target"] = y
    X = GenericDataLoader(Xraw)

    for retry in range(2):
        test_plugin = plugin(
            n_iter=5000,
            compress_dataset=compress_dataset,
        )
        evaluator = PerformanceEvaluatorXGB()

        test_plugin.fit(X)
        X_syn = test_plugin.generate()

        results.append(evaluator.evaluate(X, X_syn)["syn_id"])

    print(plugin.name(), compress_dataset, results)
    assert np.mean(results) > 0.7


@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
def test_plugin_conditional_ctgan() -> None:
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


def gen_datetime(min_year: int = 2000, max_year: int = datetime.now().year) -> datetime:
    # generate a datetime in format yyyy-mm-dd hh:mm:ss.000000
    start = datetime(min_year, 1, 1, 00, 00, 00)
    years = max_year - min_year + 1
    end = start + timedelta(days=365 * years)
    return start + (end - start) * random.random()


@pytest.mark.slow
def test_plugin_encoding() -> None:
    data = [[gen_datetime(), i % 2 == 0, i] for i in range(1000)]

    df = pd.DataFrame(data, columns=["date", "bool", "int"])
    test_plugin = plugin(generator_n_units_hidden=5)
    test_plugin.fit(df)

    syn = test_plugin.generate(10)

    assert len(syn) == 10
    assert test_plugin.schema_includes(syn)

    syn_df = syn.dataframe()

    assert syn_df["date"].infer_objects().dtype.kind == "M"
    assert syn_df["bool"].infer_objects().dtype.kind == "b"
