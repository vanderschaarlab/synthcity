# stdlib
import random
from datetime import datetime, timedelta

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
from synthcity.plugins.privacy.plugin_aim import plugin
from synthcity.utils.datasets.categorical.categorical_adult import (
    CategoricalAdultDataloader,
)
from synthcity.utils.serialization import load, save

plugin_name = "aim"
plugin_args = {
    "epsilon": 1.0,
    "delta": 1e-9,
    "max_model_size": 80,
    "degree": 2,
    "num_marginals": None,
    "max_cells": 1000,
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
    assert len(test_plugin.hyperparameter_space()) == 6


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
def test_plugin_fit(test_plugin: Plugin) -> None:
    X = CategoricalAdultDataloader().load().head()
    test_plugin.fit(GenericDataLoader(X))


@pytest.mark.slow
@pytest.mark.parametrize(
    "test_plugin",
    generate_fixtures(plugin_name, plugin, plugin_args),
)
@pytest.mark.parametrize("serialize", [True, False])
def test_plugin_generate(test_plugin: Plugin, serialize: bool) -> None:
    X = CategoricalAdultDataloader().load().head()
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


@pytest.mark.slow
@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
def test_plugin_generate_constraints_aim(test_plugin: Plugin) -> None:
    X = CategoricalAdultDataloader().load().head()
    test_plugin.fit(GenericDataLoader(X, target_column="income>50K"))

    constraints = Constraints(
        rules=[
            ("income>50K", "eq", 1),
        ]
    )

    X_gen = test_plugin.generate(constraints=constraints).dataframe()
    assert len(X_gen) == len(X)
    assert test_plugin.schema_includes(X_gen)
    assert constraints.filter(X_gen).sum() == len(X_gen)
    assert (X_gen["income>50K"] == 1).all()

    X_gen = test_plugin.generate(count=50, constraints=constraints).dataframe()
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)
    assert constraints.filter(X_gen).sum() == len(X_gen)
    assert list(X_gen.columns) == list(X.columns)


def test_sample_hyperparams() -> None:
    assert plugin is not None
    for i in range(100):
        args = plugin.sample_hyperparameters()
        assert plugin(**args) is not None


@pytest.mark.slow
@pytest.mark.parametrize("compress_dataset", [True, False])
def test_eval_performance_aim(compress_dataset: bool) -> None:
    assert plugin is not None
    results = []

    X_raw, y = load_iris(as_frame=True, return_X_y=True)
    X_raw["target"] = y
    # Descretize the data
    num_bins = 3
    for col in X_raw.columns:
        X_raw[col] = pd.cut(X_raw[col], bins=num_bins, labels=list(range(num_bins)))

    X = GenericDataLoader(X_raw, target_column="target")

    for retry in range(2):
        test_plugin = plugin(**plugin_args)
        evaluator = PerformanceEvaluatorXGB()

        test_plugin.fit(X)
        X_syn = test_plugin.generate(count=1000)

        results.append(evaluator.evaluate(X, X_syn)["syn_id"])
        print(results)
    assert np.mean(results) > 0.7


def gen_datetime(min_year: int = 2000, max_year: int = datetime.now().year) -> datetime:
    # generate a datetime in format yyyy-mm-dd hh:mm:ss.000000
    start = datetime(min_year, 1, 1, 00, 00, 00)
    years = max_year - min_year + 1
    end = start + timedelta(days=365 * years)
    return start + (end - start) * random.random()


def test_plugin_encoding() -> None:
    assert plugin is not None
    data = [[gen_datetime(), i % 2 == 0, i] for i in range(10)]

    df = pd.DataFrame(data, columns=["date", "bool", "int"])
    X = GenericDataLoader(df)
    test_plugin = plugin(**plugin_args)
    test_plugin.fit(X)

    syn = test_plugin.generate(10)

    assert len(syn) == 10
    assert test_plugin.schema_includes(syn)

    syn_df = syn.dataframe()

    assert syn_df["date"].infer_objects().dtype.kind == "M"
    assert syn_df["bool"].infer_objects().dtype.kind == "b"
