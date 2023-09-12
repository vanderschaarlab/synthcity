# third party
import pytest
from lifelines.datasets import load_rossi
from surv_helpers import generate_fixtures

# synthcity absolute
from synthcity.plugins import Plugin
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader
from synthcity.plugins.generic.plugin_marginal_distributions import plugin

X = load_rossi()
data = SurvivalAnalysisDataLoader(
    X,
    target_column="arrest",
    time_to_event_column="week",
)


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
    test_plugin.fit(data)


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures("marginal_distributions", plugin)
)
def test_plugin_generate(test_plugin: Plugin) -> None:
    test_plugin.fit(data)

    X_gen = test_plugin.generate().dataframe()
    assert len(X_gen) == len(X)
    assert test_plugin.schema_includes(X_gen)

    X_gen = test_plugin.generate(50).dataframe()
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)
    assert list(X_gen.columns) == list(X.columns)
