# third party
import pytest
from lifelines.datasets import load_rossi
from surv_helpers import generate_fixtures

# synthcity absolute
from synthcity.plugins import Plugin
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader
from synthcity.plugins.survival_analysis.plugin_survae import plugin

X = load_rossi()
data = SurvivalAnalysisDataLoader(
    X,
    target_column="arrest",
    time_to_event_column="week",
)


plugin_name = "survae"
plugins_args = {
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
    assert len(test_plugin.hyperparameter_space()) == 13


@pytest.mark.parametrize(
    "dataloader_sampling_strategy",
    [
        "imbalanced_censoring",
        "imbalanced_time_censoring",
        "none",
    ],
)
@pytest.mark.parametrize(
    "tte_strategy",
    [
        "survival_function",
        "uncensoring",
    ],
)
@pytest.mark.slow
def test_plugin_fit(dataloader_sampling_strategy: str, tte_strategy: str) -> None:
    test_plugin = plugin(
        tte_strategy=tte_strategy,
        dataloader_sampling_strategy=dataloader_sampling_strategy,
        device="cpu",
        **plugins_args
    )

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
@pytest.mark.slow
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
