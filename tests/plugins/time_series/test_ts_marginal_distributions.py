# third party
import pytest
from ts_helpers import generate_fixtures

# synthcity absolute
from synthcity.plugins import Plugin
from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
from synthcity.plugins.generic.plugin_marginal_distributions import plugin
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader

static_data, temporal_data, temporal_horizons, outcome = GoogleStocksDataloader().load()
data = TimeSeriesDataLoader(
    temporal_data=temporal_data,
    temporal_horizons=temporal_horizons,
    static_data=static_data,
    outcome=outcome,
)


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
    assert len(X_gen) == len(data)
    assert test_plugin.schema_includes(X_gen)

    X_gen = test_plugin.generate(50).dataframe()
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)
    assert list(X_gen.columns) == list(data.columns)
