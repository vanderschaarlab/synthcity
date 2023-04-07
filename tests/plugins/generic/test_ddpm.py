# stdlib
from itertools import product
from typing import Any, Generator

# third party
import numpy as np
import pandas as pd
import pytest
from generic_helpers import generate_fixtures
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics.eval import PerformanceEvaluatorXGB
from synthcity.plugins import Plugin
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins.generic.plugin_ddpm import plugin

plugin_name = "ddpm"
plugin_params = dict(
    n_iter=500,
    batch_size=1000,
    num_timesteps=100,
    model_type="mlp",
)


def extend_fixtures(
    plugin_name: str = plugin_name,
    plugin: Any = plugin,
    plugin_params: dict = plugin_params,
    **extra_params: list
) -> Generator:
    if not extra_params:
        yield from generate_fixtures(plugin_name, plugin, plugin_params)
        return
    param_set = list(product(*extra_params.values()))
    for values in param_set:
        params = plugin_params.copy()
        params.update(zip(extra_params.keys(), values))
        yield from generate_fixtures(plugin_name, plugin, params)


@pytest.mark.parametrize("test_plugin", extend_fixtures())
def test_plugin_sanity(test_plugin: Plugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", extend_fixtures())
def test_plugin_name(test_plugin: Plugin) -> None:
    assert test_plugin.name() == plugin_name


@pytest.mark.parametrize("test_plugin", extend_fixtures())
def test_plugin_type(test_plugin: Plugin) -> None:
    assert test_plugin.type() == "generic"


@pytest.mark.parametrize(
    "test_plugin", extend_fixtures(is_classification=[True, False])
)
def test_plugin_fit(test_plugin: Plugin) -> None:
    X = pd.DataFrame(load_iris()["data"])
    test_plugin.fit(GenericDataLoader(X))


@pytest.mark.parametrize(
    "test_plugin", extend_fixtures(is_classification=[True, False])
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


@pytest.mark.parametrize(
    "test_plugin", extend_fixtures(is_classification=[True, False])
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


@pytest.mark.parametrize("test_plugin", extend_fixtures())
def test_plugin_hyperparams(test_plugin: Plugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 4


def test_sample_hyperparams() -> None:
    for i in range(100):
        args = plugin.sample_hyperparameters()
        assert plugin(**args) is not None


@pytest.mark.slow
@pytest.mark.parametrize("compress_dataset", [True, False])
def test_eval_performance_ddpm(compress_dataset: bool) -> None:
    results = []

    Xraw, y = load_iris(return_X_y=True, as_frame=True)
    Xraw["target"] = y
    X = GenericDataLoader(Xraw)

    for _ in range(2):
        test_plugin = plugin(**plugin_params, compress_dataset=compress_dataset)
        evaluator = PerformanceEvaluatorXGB()

        test_plugin.fit(X)
        X_syn = test_plugin.generate()

        results.append(evaluator.evaluate(X, X_syn)["syn_id"])

    print(plugin.name(), results)
    assert np.mean(results) > 0.8
