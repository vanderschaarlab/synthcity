# stdlib
import sys

# third party
import pytest
from lifelines.datasets import load_rossi
from surv_helpers import generate_fixtures

# synthcity absolute
from synthcity.plugins import Plugin
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader
from synthcity.plugins.survival_analysis.plugin_survival_ctgan import plugin

X = load_rossi()
data = SurvivalAnalysisDataLoader(
    X,
    target_column="arrest",
    time_to_event_column="week",
)


plugin_name = "survival_ctgan"
plugins_args = {
    "generator_n_layers_hidden": 1,
    "generator_n_units_hidden": 10,
    "uncensoring_model": "cox_ph",
    "n_iter": 100,
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
    assert len(test_plugin.hyperparameter_space()) == 14


@pytest.mark.parametrize(
    "tte_strategy",
    [
        "survival_function",
        "uncensoring",
    ],
)
def test_plugin_fit(tte_strategy: str) -> None:
    test_plugin = plugin(tte_strategy=tte_strategy, device="cpu", **plugins_args)

    test_plugin.fit(data)


@pytest.mark.parametrize("strategy", ["uncensoring", "survival_function"])
def test_plugin_generate(strategy: str) -> None:
    test_plugin = plugin(tte_strategy=strategy, **plugins_args)

    test_plugin.fit(data)

    X_gen = test_plugin.generate()
    assert len(X_gen) == len(X)
    assert test_plugin.schema_includes(X_gen)

    X_gen = test_plugin.generate(50)
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)


@pytest.mark.parametrize("strategy", ["uncensoring", "survival_function"])
def test_survival_plugin_generate_constraints(strategy: str) -> None:
    test_plugin = plugin(tte_strategy=strategy, **plugins_args)

    test_plugin.fit(data)

    constraints = Constraints(
        rules=[
            ("week", "le", 40),
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


@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
def test_plugin_generate_with_conditional() -> None:
    bin_conditional = X["wexp"]

    test_plugin = plugin()

    test_plugin.fit(data, cond=bin_conditional)

    # generate using training conditional
    X_gen = test_plugin.generate(2 * len(X))
    assert len(X_gen) == 2 * len(X)
    assert test_plugin.schema_includes(X_gen)

    # generate using custom conditional
    count = 100
    gen_cond = [1] * count
    X_gen = test_plugin.generate(count, cond=gen_cond)
    assert X_gen["wexp"].sum() > 80  # at least 80% samples respect the conditional
