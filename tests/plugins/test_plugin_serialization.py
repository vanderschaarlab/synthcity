# stdlib
from typing import Any

# third party
import pandas as pd
import pytest
from lifelines.datasets import load_rossi
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.plugins import Plugin, Plugins
from synthcity.plugins.core.dataloader import (
    SurvivalAnalysisDataLoader,
    TimeSeriesDataLoader,
)
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
from synthcity.utils.serialization import load, save
from synthcity.version import MAJOR_VERSION


def test_version() -> None:
    for plugin in Plugins().list():
        assert Plugins().get(plugin).version() == MAJOR_VERSION


def test_invalid_version() -> None:
    for plugin in Plugins().list():
        buff = Plugins().get(plugin).save()
        decoded = load(buff)
        decoded["version"] = "invalid"
        invalid_buff = save(decoded)

        with pytest.raises(RuntimeError):
            Plugin.load(invalid_buff)

        with pytest.raises(RuntimeError):
            Plugins().load(invalid_buff)


def sanity_check(original: Any, reloaded: Any, generate: bool = False) -> None:
    assert reloaded.name() == original.name()

    for key in original.__dict__:
        assert key in reloaded.__dict__


def verify_serialization(model: Any, generate: bool = False) -> None:
    # pickle test
    buff = save(model)

    reloaded = load(buff)
    sanity_check(model, reloaded, generate=generate)

    # API test
    buff = model.save()

    reloaded = Plugin.load(buff)
    sanity_check(model, reloaded, generate=generate)

    reloaded = Plugins().load(buff)
    sanity_check(model, reloaded, generate=generate)


def test_serialization_sanity() -> None:
    generic_data = pd.DataFrame(load_iris()["data"])
    plugins = Plugins(categories=["privacy"])

    # pre-training
    syn_model = plugins.get("adsgan", strict=False, n_iter=10)
    verify_serialization(syn_model)

    # post-training
    syn_model.fit(generic_data)
    verify_serialization(syn_model, generate=True)


@pytest.mark.parametrize("plugin", Plugins(categories=["privacy"]).list())
@pytest.mark.slow
def test_serialization_privacy_plugins(plugin: str) -> None:
    generic_data = pd.DataFrame(load_iris()["data"])
    plugins = Plugins(categories=["privacy"])

    # pre-training
    syn_model = plugins.get(plugin, strict=False)
    verify_serialization(syn_model)

    # post-training
    syn_model.fit(generic_data)
    verify_serialization(syn_model, generate=True)


@pytest.mark.parametrize("plugin", Plugins(categories=["generic"]).list())
@pytest.mark.slow
def test_serialization_generic_plugins(plugin: str) -> None:
    generic_data = pd.DataFrame(load_iris()["data"])
    plugins = Plugins(categories=["generic"])

    # pre-training
    syn_model = plugins.get(plugin, strict=False)
    verify_serialization(syn_model)

    # post-training
    syn_model.fit(generic_data)
    verify_serialization(syn_model, generate=True)


@pytest.mark.parametrize("plugin", Plugins(categories=["time_series"]).list())
@pytest.mark.slow
def test_serialization_ts_plugins(plugin: str) -> None:
    (
        static_data,
        temporal_data,
        observation_times,
        outcome,
    ) = GoogleStocksDataloader().load()
    ts_data = TimeSeriesDataLoader(
        temporal_data=temporal_data,
        observation_times=observation_times,
        static_data=static_data,
        outcome=outcome,
    )

    ts_plugins = Plugins(categories=["time_series"])
    syn_model = ts_plugins.get(plugin, n_iter=10, strict=False)

    # pre-training
    verify_serialization(syn_model)

    # post-training
    syn_model.fit(ts_data)
    verify_serialization(syn_model, generate=True)


@pytest.mark.parametrize("plugin", ["survival_gan"])
@pytest.mark.slow
def test_serialization_surv_plugins(plugin: str) -> None:
    X = load_rossi()
    surv_data = SurvivalAnalysisDataLoader(
        X,
        target_column="arrest",
        time_to_event_column="week",
    )
    surv_plugins = Plugins(categories=["survival_analysis"])
    syn_model = surv_plugins.get(plugin, n_iter=10, strict=False)

    # pre-training
    verify_serialization(syn_model)

    # post-training
    syn_model.fit(surv_data)
    verify_serialization(syn_model)
