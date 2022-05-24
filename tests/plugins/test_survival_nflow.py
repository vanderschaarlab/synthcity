# third party
import pytest
from helpers import generate_fixtures
from lifelines.datasets import load_rossi

# synthcity absolute
from synthcity.plugins import Plugin
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.plugin_survival_nflow import plugin

plugin_name = "survival_nflow"
plugins_args = {
    "n_iter": 10,
    "uncensoring_model": "cox_ph",
}


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugins_args)
)
def test_plugin_sanity(test_plugin: Plugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugins_args)
)
def test_plugin_name(test_plugin: Plugin) -> None:
    assert test_plugin.name() == plugin_name


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugins_args)
)
def test_plugin_type(test_plugin: Plugin) -> None:
    assert test_plugin.type() == "survival_analysis"


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugins_args)
)
def test_plugin_hyperparams(test_plugin: Plugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 9


@pytest.mark.parametrize("strategy", ["uncensoring", "survival_function"])
def test_plugin_fit(strategy: str) -> None:
    test_plugin = plugin(
        target_column="arrest",
        time_to_event_column="week",
        strategy=strategy,
        **plugins_args
    )

    X = load_rossi()
    test_plugin.fit(X)


@pytest.mark.parametrize("strategy", ["uncensoring", "survival_function"])
def test_plugin_generate(strategy: str) -> None:
    test_plugin = plugin(
        target_column="arrest",
        time_to_event_column="week",
        strategy=strategy,
        **plugins_args
    )

    X = load_rossi()
    test_plugin.fit(X)

    X_gen = test_plugin.generate()
    assert len(X_gen) == len(X)
    assert test_plugin.schema_includes(X_gen)

    X_gen = test_plugin.generate(50)
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)


@pytest.mark.parametrize("strategy", ["uncensoring", "survival_function"])
def test_survival_plugin_generate_constraints(strategy: str) -> None:
    test_plugin = plugin(
        target_column="arrest",
        time_to_event_column="week",
        strategy=strategy,
        **plugins_args
    )

    X = load_rossi()
    test_plugin.fit(X)

    constraints = Constraints(
        rules=[
            ("week", "le", 40),
        ]
    )

    X_gen = test_plugin.generate(constraints=constraints)
    assert len(X_gen) == len(X)
    assert test_plugin.schema_includes(X_gen)
    assert constraints.filter(X_gen).sum() == len(X_gen)

    X_gen = test_plugin.generate(count=50, constraints=constraints)
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)
    assert constraints.filter(X_gen).sum() == len(X_gen)
    assert list(X_gen.columns) == list(X.columns)


def test_sample_hyperparams() -> None:
    for i in range(100):
        args = plugin.sample_hyperparameters()

        assert plugin(**args) is not None
