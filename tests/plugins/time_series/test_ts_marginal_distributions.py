# stdlib
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
from synthcity.plugins.generic.plugin_marginal_distributions import plugin
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
from synthcity.utils.datasets.time_series.pbc import PBCDataloader
from synthcity.utils.datasets.time_series.sine import SineDataloader


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures("marginal_distributions", plugin)
)
def test_plugin_fit(test_plugin: Plugin) -> None:
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

    test_plugin.fit(data)


@pytest.mark.parametrize(
    "test_plugin", [generate_fixtures("marginal_distributions", plugin)[0]]
)
@pytest.mark.parametrize(
    "source",
    [
        SineDataloader(with_missing=False),
        SineDataloader(with_missing=True),
        GoogleStocksDataloader(),
    ],
)
def test_plugin_generate(test_plugin: Plugin, source: Any) -> None:
    static_data, temporal_data, observation_times, outcome = source.load()
    data = TimeSeriesDataLoader(
        temporal_data=temporal_data,
        observation_times=observation_times,
        static_data=static_data,
        outcome=outcome,
    )

    test_plugin.fit(data)

    X_gen = test_plugin.generate().dataframe()
    assert len(X_gen) == len(data)
    assert test_plugin.schema_includes(X_gen)

    X_gen = test_plugin.generate(50).dataframe()
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)
    assert list(X_gen.columns) == list(data.columns)


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

    test_plugin = plugin()
    test_plugin.fit(survival_data)

    X_gen = test_plugin.generate(10)

    assert X_gen.type() == "time_series_survival"
    assert len(X_gen) == 10
    assert test_plugin.schema_includes(X_gen)
    assert list(X_gen.columns) == list(survival_data.columns)
