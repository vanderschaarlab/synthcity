# stdlib
from typing import Any

# third party
import pytest
from lifelines.datasets import load_rossi
from sklearn.datasets import load_breast_cancer

# synthcity absolute
from synthcity.plugins.core.dataloader import (
    GenericDataLoader,
    SurvivalAnalysisDataLoader,
    TimeSeriesDataLoader,
)
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
from synthcity.utils.datasets.time_series.sine import SineDataloader


def test_generic_dataloader() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    X["target"] = y

    loader = GenericDataLoader(X, target_column="target")

    assert loader.raw().shape == X.shape
    assert loader.type() == "generic"
    assert loader.shape == X.shape
    assert list(loader.columns) == list(X.columns)
    assert (loader.numpy() == X.values).all()
    assert (loader.dataframe().values == X.values).all()
    assert (loader.values == X.values).all()
    assert len(loader) == len(X)

    assert loader.info()["data_type"] == "generic"
    assert loader.info()["len"] == len(X)
    assert loader.info()["static_features"] == list(X.columns)

    Xu, yu = loader.unpack()
    assert Xu.shape == (len(X), X.shape[1] - 1)
    assert yu.shape == y.shape

    assert loader.sample(5).shape == (5, X.shape[1])
    assert len(loader[X.columns[0]]) == len(X)
    assert loader.train().shape == (455, X.shape[1])
    assert loader.test().shape == (114, X.shape[1])
    assert loader.hash() != ""


def test_survival_dataloader() -> None:
    df = load_rossi()

    target_column = "arrest"
    time_to_event_column = "week"

    X = df.drop(columns=[target_column, time_to_event_column])
    T = df[time_to_event_column]
    E = df[target_column]

    loader = SurvivalAnalysisDataLoader(
        df,
        time_to_event_column=time_to_event_column,
        target_column=target_column,
        time_horizons=[20],
    )

    assert (loader.dataframe().values == df.values).all()
    assert loader.raw().shape == df.shape
    assert loader.unpack()[0].shape == X.shape
    assert loader.unpack()[1].shape == T.shape
    assert loader.unpack()[2].shape == E.shape

    assert loader.type() == "survival_analysis"
    assert loader.shape == df.shape
    assert sorted(list(loader.dataframe().columns)) == sorted(list(df.columns))
    assert (loader.numpy() == df.values).all()
    assert len(loader) == len(X)

    assert loader.info()["data_type"] == "survival_analysis"
    assert loader.info()["len"] == len(X)
    assert loader.info()["static_features"] == list(df.columns)

    assert loader.sample(5).shape == (5, df.shape[1])
    assert len(loader[X.columns[0]]) == len(X)
    assert loader.train().shape == (345, df.shape[1])
    assert loader.test().shape == (87, df.shape[1])
    assert loader.hash() != ""


@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_time_series_dataloader(source: Any) -> None:
    static_data, temporal_data, outcome = source().load()

    loader = TimeSeriesDataLoader(
        temporal_data=temporal_data,
        static_data=static_data,
        outcome=outcome,
        train_size=0.8,
    )

    assert len(loader.raw()) == 4
    feat_cnt = temporal_data[0].shape[0] * temporal_data[0].shape[1]
    if static_data is not None:
        feat_cnt += static_data.shape[1]
    if outcome is not None:
        feat_cnt += outcome.shape[1]

    assert loader.dataframe().shape == (len(temporal_data), feat_cnt)
    assert loader.numpy().shape == (len(temporal_data), feat_cnt)
    assert loader.temporal_numpy().shape == (
        len(temporal_data),
        temporal_data[0].shape[0],
        temporal_data[0].shape[1],
    )
    assert (loader.temporal_columns == temporal_data[0].columns).all()
    assert len(loader) == len(temporal_data)

    info = loader.info()
    assert info["data_type"] == "time_series"
    assert info["len"] == len(temporal_data)
    assert info["static_features"] == (
        list(static_data.columns) if static_data is not None else []
    )
    assert info["temporal_features"] == list(temporal_data[0].columns)
    assert info["outcome_features"] == (
        list(outcome.columns) if outcome is not None else []
    )
    assert info["seq_len"] == len(temporal_data[0])
    assert info["sensitive_features"] == []

    train_len = int(0.8 * (len(temporal_data)))
    assert loader.train().shape == (train_len, feat_cnt)
    assert loader.test().shape == (len(temporal_data) - train_len, feat_cnt)

    assert loader.hash() != ""


@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_time_series_pack_unpack(source: Any) -> None:
    static_data, temporal_data, outcome = source().load()

    loader = TimeSeriesDataLoader(
        temporal_data=temporal_data,
        static_data=static_data,
        outcome=outcome,
    )

    unp_static_data, unp_temporal, unp_out = TimeSeriesDataLoader.unpack_raw_data(
        loader.dataframe(),
        loader.static_features,
        loader.temporal_features,
        loader.outcome_features,
        loader.seq_len,
    )

    if static_data is not None:
        assert unp_static_data.shape == static_data.shape
        assert (unp_static_data.values == static_data.values).all()
        assert (unp_static_data.columns == static_data.columns).all()
    else:
        assert unp_static_data is None

    if outcome is not None:
        assert unp_out.shape == outcome.shape
        assert (unp_out.values == outcome.values).all()
        assert (unp_out.columns == outcome.columns).all()
    else:
        assert unp_out is None

    assert len(unp_temporal) == len(temporal_data)
    for idx, item in enumerate(temporal_data):
        assert unp_temporal[idx].shape == temporal_data[idx].shape
        assert (unp_temporal[idx].columns == temporal_data[idx].columns).all()
        assert (unp_temporal[idx].values == temporal_data[idx].values).all()
