# third party
import pandas as pd
import pytest
from generic_helpers import generate_fixtures
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.plugins import Plugin
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins.generic.plugin_marginal_distributions import plugin


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures("marginal_distributions", plugin)
)
def test_plugin_sanity(test_plugin: Plugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures("marginal_distributions", plugin)
)
def test_plugin_name(test_plugin: Plugin) -> None:
    assert test_plugin.name() == "marginal_distributions"


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures("marginal_distributions", plugin)
)
def test_plugin_type(test_plugin: Plugin) -> None:
    assert test_plugin.type() == "debug"


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures("marginal_distributions", plugin)
)
def test_plugin_hyperparams(test_plugin: Plugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 0


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures("marginal_distributions", plugin)
)
def test_plugin_fit(test_plugin: Plugin) -> None:
    X = pd.DataFrame(load_iris()["data"])
    test_plugin.fit(GenericDataLoader(X))


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures("marginal_distributions", plugin)
)
def test_plugin_generate(test_plugin: Plugin) -> None:
    X = pd.DataFrame(load_iris()["data"])
    test_plugin.fit(GenericDataLoader(X))

    X_gen = test_plugin.generate().dataframe()
    assert len(X_gen) == len(X)
    assert test_plugin.schema_includes(X_gen)

    X_gen = test_plugin.generate(50).dataframe()
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)
    assert list(X_gen.columns) == list(X.columns)


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures("marginal_distributions", plugin)
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
