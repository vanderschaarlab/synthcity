# stdlib
import sys
from typing import Any

# third party
import numpy as np
import pytest
from ts_helpers import generate_fixtures

# synthcity absolute
from synthcity.plugins import Plugin
from synthcity.plugins.core.dataloader import (
    TimeSeriesDataLoader,
    TimeSeriesSurvivalDataLoader,
)
from synthcity.plugins.time_series.plugin_fflows import plugin
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
from synthcity.utils.datasets.time_series.pbc import PBCDataloader
from synthcity.utils.datasets.time_series.sine import SineDataloader

plugin_name = "fflows"


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
    assert len(test_plugin.hyperparameter_space()) == 10


@pytest.mark.slow
def test_plugin_fit() -> None:
    (
        static_data,
        temporal_data,
        observation_times,
        outcome,
    ) = GoogleStocksDataloader().load()
    data = TimeSeriesDataLoader(
        temporal_data=temporal_data,
        observation_times=observation_times,
        static_data=static_data,
        outcome=outcome,
    )

    test_plugin = plugin(
        n_iter=10,
    )

    test_plugin.fit(data)


@pytest.mark.parametrize(
    "source",
    [
        SineDataloader(with_missing=True),
        GoogleStocksDataloader(),
    ],
)
@pytest.mark.slow
def test_plugin_generate(source: Any) -> None:
    static_data, temporal_data, observation_times, outcome = source.load()
    data = TimeSeriesDataLoader(
        temporal_data=temporal_data,
        observation_times=observation_times,
        static_data=static_data,
        outcome=outcome,
    )
    test_plugin = plugin(
        n_iter=10,
    )
    test_plugin.fit(data)

    X_gen = test_plugin.generate(50)
    assert len(X_gen.ids()) == 50
    assert test_plugin.schema_includes(X_gen)
    assert list(X_gen.columns) == list(data.columns)


def test_sample_hyperparams() -> None:
    for i in range(100):
        args = plugin.sample_hyperparameters()

        assert plugin(**args) is not None


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="test only with python3.9 or higher"
)
@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
@pytest.mark.slow
def test_plugin_generate_survival() -> None:
    (
        static_surv,
        temporal_surv,
        temporal_surv_horizons,
        outcome_surv,
    ) = PBCDataloader().load()
    T, E = outcome_surv

    horizons = [0.25, 0.5, 0.75]
    time_horizons = np.quantile(T, horizons).tolist()

    survival_data = TimeSeriesSurvivalDataLoader(
        temporal_data=temporal_surv,
        observation_times=temporal_surv_horizons,
        static_data=static_surv,
        T=T,
        E=E,
        time_horizons=time_horizons,
    )

    test_plugin = plugin(
        n_iter=10,
    )
    test_plugin.fit(survival_data)

    X_gen = test_plugin.generate(10)

    assert X_gen.type() == "time_series_survival"
    assert len(X_gen.ids()) == 10
    assert test_plugin.schema_includes(X_gen)
    assert list(X_gen.columns) == list(survival_data.columns)
