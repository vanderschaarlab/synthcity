# stdlib
import sys
from typing import Any

# third party
import numpy as np
import pandas as pd
import pytest
from ts_helpers import generate_fixtures

# synthcity absolute
from synthcity.plugins import Plugin
from synthcity.plugins.core.dataloader import (
    TimeSeriesDataLoader,
    TimeSeriesSurvivalDataLoader,
)
from synthcity.plugins.time_series.plugin_timegan import plugin
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
from synthcity.utils.datasets.time_series.pbc import PBCDataloader
from synthcity.utils.datasets.time_series.sine import SineDataloader

plugin_name = "timegan"


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
    assert len(test_plugin.hyperparameter_space()) == 18


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


@pytest.mark.parametrize(
    "source", [SineDataloader(with_missing=True), GoogleStocksDataloader()]
)
def test_plugin_generate_static_cond(source: Any) -> None:
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

    cnt = 50
    static_seed = static_data.head(1).values.tolist() * cnt
    X_gen = test_plugin.generate(
        cnt, static_data=pd.DataFrame(static_seed, columns=static_data.columns)
    )
    static_gen, _, _, _ = X_gen.unpack(as_numpy=True)
    assert len(X_gen.ids()) == cnt
    assert test_plugin.schema_includes(X_gen)
    assert list(X_gen.columns) == list(data.columns)
    assert np.isclose(np.asarray(static_gen), np.asarray(static_seed)).all()


@pytest.mark.parametrize(
    "source",
    [
        SineDataloader(with_missing=True),
        GoogleStocksDataloader(),
    ],
)
def test_plugin_generate_horizons_cond(source: Any) -> None:
    static_data, temporal_data, observation_times, outcome = source.load()
    data = TimeSeriesDataLoader(
        temporal_data=temporal_data,
        observation_times=observation_times,
        static_data=static_data,
        outcome=outcome,
    )
    _, _, observation_times, _ = data.unpack(pad=True)
    test_plugin = plugin(
        n_iter=10,
    )
    test_plugin.fit(data)

    cnt = 50
    horizon_seed = []
    for r in range(cnt):
        horizon_seed.append(observation_times[0])

    X_gen = test_plugin.generate(cnt, observation_times=horizon_seed)
    assert len(X_gen.ids()) == cnt
    assert test_plugin.schema_includes(X_gen)
    assert list(X_gen.columns) == list(data.columns)


def test_sample_hyperparams() -> None:
    for i in range(100):
        args = plugin.sample_hyperparameters()

        assert plugin(**args) is not None


@pytest.mark.slow
def test_timegan_plugin_generate_survival() -> None:
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
    st, tmp, tmp_horiz, genT, genE = X_gen.unpack()

    assert X_gen.type() == "time_series_survival"
    assert len(X_gen.ids()) == 10
    assert test_plugin.schema_includes(X_gen)
    assert list(X_gen.columns) == list(survival_data.columns)


@pytest.mark.parametrize(
    "sampling_strategy", ["none", "imbalanced_censoring", "imbalanced_time_censoring"]
)
@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="test only with python3.9 or higher"
)
@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
def test_plugin_generate_survival_sampler(sampling_strategy: str) -> None:
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

    test_plugin = plugin(n_iter=10, dataloader_sampling_strategy=sampling_strategy)
    assert test_plugin.dataloader_sampling_strategy == sampling_strategy
    test_plugin.fit(survival_data)

    X_gen = test_plugin.generate(10)
    st, tmp, tmp_horiz, genT, genE = X_gen.unpack()

    assert X_gen.type() == "time_series_survival"
    assert len(X_gen.ids()) == 10
    assert test_plugin.schema_includes(X_gen)
    assert list(X_gen.columns) == list(survival_data.columns)
