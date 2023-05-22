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
from synthcity.metrics.weighted_metrics import WeightedMetrics
from synthcity.plugins import Plugin
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins.generic.plugin_ddpm import plugin
from synthcity.utils.callbacks import EarlyStopping

plugin_name = "ddpm"
plugin_params = dict(
    n_iter=1000,
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


def test_plugin_early_stop() -> None:
    X = pd.DataFrame(load_iris()["data"])
    early_stop = EarlyStopping(patience=10, min_epochs=50)
    test_plugin = plugin(
        validation_size=0.2,
        validation_metric=WeightedMetrics(
            metrics=[("detection", "detection_xgb")],
            weights=[1],
        ),
        callbacks=[early_stop],
        **plugin_params
    )
    test_plugin.fit(GenericDataLoader(X))
    n_epochs = len(test_plugin.validation_history)
    assert n_epochs >= early_stop.min_epochs
    if n_epochs > early_stop.min_epochs:
        assert early_stop.best_epoch == n_epochs - early_stop.patience - 1
    assert (
        early_stop.best_score == test_plugin.validation_history[early_stop.best_epoch]
    )
    assert early_stop.best_score == min(test_plugin.validation_history)


test_plugin_early_stop()


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
    X, y = load_iris(as_frame=True, return_X_y=True)
    X["target"] = y
    test_plugin.fit(GenericDataLoader(X))

    constraints = Constraints(
        rules=[
            ("target", "eq", 1),
            ("sepal length (cm)", "le", 6),
            ("sepal length (cm)", "ge", 4.3),
            ("sepal width (cm)", "le", 4.4),
            ("sepal width (cm)", "ge", 3),
            ("petal length (cm)", "le", 5.5),
            ("petal length (cm)", "ge", 1.0),
            ("petal width (cm)", "le", 2),
            ("petal width (cm)", "ge", 0.1),
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


@pytest.mark.parametrize("test_plugin", extend_fixtures())
def test_plugin_hyperparams(test_plugin: Plugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 4


def test_sample_hyperparams() -> None:
    for _ in range(100):
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
