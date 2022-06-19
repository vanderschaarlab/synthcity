# third party
import pytest
from ts_helpers import generate_fixtures

# synthcity absolute
from synthcity.plugins import Plugin
from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
from synthcity.plugins.time_series.plugin_probabilistic_ar import plugin
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader

static_data, temporal_data, temporal_horizons, outcome = GoogleStocksDataloader().load()
data = TimeSeriesDataLoader(
    temporal_data=temporal_data,
    temporal_horizons=temporal_horizons,
    static_data=static_data,
    outcome=outcome,
)
plugin_name = "probabilistic_ar"


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_sanity(test_plugin: Plugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_name(test_plugin: Plugin) -> None:
    assert test_plugin.name() == plugin_name


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_type(test_plugin: Plugin) -> None:
    assert test_plugin.type() == "time_series"


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_hyperparams(test_plugin: Plugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 2


def test_plugin_fit() -> None:
    test_plugin = plugin(
        n_iter=10,
    )

    test_plugin.fit(data)


def test_plugin_generate() -> None:
    test_plugin = plugin(
        n_iter=10,
    )
    test_plugin.fit(data)

    X_gen = test_plugin.generate()
    assert len(X_gen) == len(data)
    assert test_plugin.schema_includes(X_gen)

    X_gen = test_plugin.generate(50)
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)
    assert list(X_gen.columns) == list(data.columns)


def test_sample_hyperparams() -> None:
    for i in range(100):
        args = plugin.sample_hyperparameters()

        assert plugin(**args) is not None
