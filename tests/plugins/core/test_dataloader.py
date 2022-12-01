# stdlib
from typing import Any

# third party
import numpy as np
import pytest
from lifelines.datasets import load_rossi
from sklearn.datasets import load_breast_cancer

# synthcity absolute
from synthcity.plugins.core.dataloader import (
    GenericDataLoader,
    SurvivalAnalysisDataLoader,
    TimeSeriesDataLoader,
    TimeSeriesSurvivalDataLoader,
    create_from_info,
)
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
from synthcity.utils.datasets.time_series.pbc import PBCDataloader
from synthcity.utils.datasets.time_series.sine import SineDataloader


def test_generic_dataloader_sanity() -> None:
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

    assert loader.sample(5).shape == (5, X.shape[1])
    assert len(loader[X.columns[0]]) == len(X)
    assert loader.train().shape == (455, X.shape[1])
    assert loader.test().shape == (114, X.shape[1])
    assert loader.hash() != ""

    assert (loader["target"].values == y.values).all()
    assert "target" not in loader.drop(columns=["target"]).columns

    assert loader.compression_protected_features() == ["target"]


def test_generic_dataloader_info() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    X["target"] = y

    loader = GenericDataLoader(X, target_column="target", sensitive_features=["target"])

    assert loader.info()["data_type"] == "generic"
    assert loader.info()["len"] == len(X)
    assert loader.info()["static_features"] == list(X.columns)
    assert loader.info()["sensitive_features"] == ["target"]
    assert loader.info()["target_column"] == "target"
    assert loader.info()["outcome_features"] == ["target"]

    new_loader = GenericDataLoader.from_info(loader.dataframe(), loader.info())
    assert new_loader.shape == loader.shape
    assert new_loader.info() == loader.info()


def test_generic_dataloader_pack_unpack() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    X["target"] = y

    loader = GenericDataLoader(X, target_column="target")

    Xu, yu = loader.unpack()
    assert Xu.shape == (len(X), X.shape[1] - 1)
    assert yu.shape == y.shape


def test_generic_dataloader_domain() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X["domain"] = y

    loader = GenericDataLoader(X, domain_column="domain")

    assert loader.domain() == "domain"
    assert loader.info()["domain_column"] == "domain"


def test_generic_dataloader_compression() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X["domain"] = y

    loader = GenericDataLoader(X, domain_column="domain")

    compressed, context = loader.compress()

    assert len(compressed) == len(loader)
    assert compressed.shape[1] <= loader.shape[1]
    assert "domain" in compressed.columns

    decompressed = compressed.decompress(context)

    assert len(decompressed) == len(loader)
    assert decompressed.shape[1] == loader.shape[1]


def test_survival_dataloader_sanity() -> None:
    df = load_rossi()

    target_column = "arrest"
    time_to_event_column = "week"

    X = df.drop(columns=[target_column, time_to_event_column])
    E = df[target_column]

    loader = SurvivalAnalysisDataLoader(
        df,
        time_to_event_column=time_to_event_column,
        target_column=target_column,
        time_horizons=[20],
    )

    assert (loader.dataframe().values == df.values).all()
    assert loader.raw().shape == df.shape

    assert loader.type() == "survival_analysis"
    assert loader.shape == df.shape
    assert sorted(list(loader.dataframe().columns)) == sorted(list(df.columns))
    assert (loader.numpy() == df.values).all()
    assert len(loader) == len(X)

    assert loader.sample(5).shape == (5, df.shape[1])
    assert len(loader[X.columns[0]]) == len(X)
    assert loader.train().shape == (345, df.shape[1])
    assert loader.test().shape == (87, df.shape[1])
    assert loader.hash() != ""

    assert (loader["arrest"].values == E.values).all()
    assert X.columns[0] not in loader.drop(columns=[X.columns[0]]).columns
    assert loader.compression_protected_features() == [
        target_column,
        time_to_event_column,
    ]


def test_survival_dataloader_info() -> None:
    df = load_rossi()

    target_column = "arrest"
    time_to_event_column = "week"

    X = df.drop(columns=[target_column, time_to_event_column])

    loader = SurvivalAnalysisDataLoader(
        df,
        time_to_event_column=time_to_event_column,
        target_column=target_column,
        time_horizons=[20],
        sensitive_features=["arrest", "week"],
    )

    assert loader.info()["data_type"] == "survival_analysis"
    assert loader.info()["len"] == len(X)
    assert loader.info()["static_features"] == list(df.columns)
    assert loader.info()["sensitive_features"] == ["arrest", "week"]
    assert loader.info()["target_column"] == "arrest"
    assert loader.info()["time_to_event_column"] == "week"
    assert loader.info()["outcome_features"] == ["arrest"]
    assert loader.info()["time_horizons"] == [20]

    new_loader = SurvivalAnalysisDataLoader.from_info(loader.dataframe(), loader.info())
    assert new_loader.shape == loader.shape
    assert new_loader.info() == loader.info()


def test_survival_dataloader_pack_unpack() -> None:
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


def test_survival_dataloader_compression() -> None:
    df = load_rossi()

    target_column = "arrest"
    time_to_event_column = "week"

    loader = SurvivalAnalysisDataLoader(
        df,
        time_to_event_column=time_to_event_column,
        target_column=target_column,
        time_horizons=[20],
    )

    compressed, context = loader.compress()

    assert len(compressed) == len(loader)
    assert compressed.shape[1] <= loader.shape[1]
    assert target_column in compressed.columns
    assert time_to_event_column in compressed.columns

    decompressed = compressed.decompress(context)

    assert len(decompressed) == len(loader)
    assert decompressed.shape[1] == loader.shape[1]
    assert sorted(decompressed.columns) == sorted(loader.columns)


@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_time_series_dataloader_sanity(source: Any) -> None:
    static_data, temporal_data, temporal_horizons, outcome = source().load()

    loader = TimeSeriesDataLoader(
        temporal_data=temporal_data,
        temporal_horizons=temporal_horizons,
        static_data=static_data,
        outcome=outcome,
        train_size=0.8,
    )

    assert len(loader.raw()) == 5
    feat_cnt = temporal_data[0].shape[1] + 2  # id, time_id
    if static_data is not None:
        feat_cnt += static_data.shape[1]
    if outcome is not None:
        feat_cnt += outcome.shape[1]

    window_len = temporal_data[0].shape[0]
    data_len = len(temporal_data) * window_len

    assert loader.dataframe().shape == (data_len, feat_cnt)
    assert loader.numpy().shape == (data_len, feat_cnt)
    assert len(loader) == data_len

    train_len = int(0.8 * (len(temporal_data)))

    train_sample = loader.train().ids()
    assert len(train_sample) == train_len

    test_sample = loader.test().ids()
    assert len(test_sample) == len(temporal_data) - train_len

    rnd_sample = loader.sample(10).ids()
    assert len(rnd_sample) == 10

    assert loader.hash() != ""
    assert loader.compression_protected_features() == outcome.columns


@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_time_series_dataloader_info(source: Any) -> None:
    static_data, temporal_data, temporal_horizons, outcome = source().load()

    loader = TimeSeriesDataLoader(
        temporal_data=temporal_data,
        temporal_horizons=temporal_horizons,
        static_data=static_data,
        outcome=outcome,
        train_size=0.8,
        sensitive_features=["test"],
    )

    window_len = temporal_data[0].shape[0]

    info = loader.info()
    assert info["data_type"] == "time_series"
    assert info["len"] == len(temporal_data) * window_len
    assert info["window_len"] == window_len
    assert info["static_features"] == (
        list(static_data.columns) if static_data is not None else []
    )
    assert info["temporal_features"] == sorted(list(temporal_data[0].columns))
    assert info["outcome_features"] == (
        list(outcome.columns) if outcome is not None else []
    )
    assert info["window_len"] == len(temporal_data[0])
    assert info["sensitive_features"] == ["test"]

    new_loader = TimeSeriesDataLoader.from_info(loader.dataframe(), loader.info())
    assert new_loader.shape == loader.shape
    assert new_loader.info() == loader.info()


@pytest.mark.parametrize(
    "source",
    [
        SineDataloader(with_missing=True),
        SineDataloader(with_missing=False),
        GoogleStocksDataloader(),
    ],
)
@pytest.mark.parametrize("repack", [True, False])
def test_time_series_pack_unpack(source: Any, repack: bool) -> None:
    static_data, temporal_data, temporal_horizons, outcome = source.load()

    loader = TimeSeriesDataLoader(
        temporal_data=temporal_data,
        temporal_horizons=temporal_horizons,
        static_data=static_data,
        outcome=outcome,
    )

    if repack:
        (
            unp_static_data,
            unp_temporal,
            unp_temporal_horizons,
            unp_out,
        ) = TimeSeriesDataLoader.unpack_raw_data(
            loader.dataframe(),
            loader.info(),
        )
    else:
        unp_static_data, unp_temporal, unp_temporal_horizons, unp_out = loader.unpack()

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
        assert unp_temporal[idx].shape == item.shape
        assert len(unp_temporal_horizons[idx]) == len(temporal_horizons[idx])
        assert sorted(unp_temporal[idx].columns) == sorted(item.columns)
        cols = list(unp_temporal[idx].columns)
        assert (unp_temporal[idx].values == item[cols].values).all()


def test_time_series_survival_dataloader_sanity() -> None:
    static_data, temporal_data, temporal_horizons, outcome = PBCDataloader().load()
    T, E = outcome

    loader = TimeSeriesSurvivalDataLoader(
        temporal_data=temporal_data,
        temporal_horizons=temporal_horizons,
        static_data=static_data,
        T=T,
        E=E,
        train_size=0.8,
    )

    assert len(loader.raw()) == 5
    max_feats = max([len(t.columns) for t in temporal_data])

    feat_cnt = max_feats + 2  # id, time id
    if static_data is not None:
        feat_cnt += static_data.shape[1]
    # outcome
    feat_cnt += 2

    total_len = 0
    for item in temporal_data:
        total_len += len(item)

    assert loader.dataframe().shape == (total_len, feat_cnt)
    assert loader.numpy().shape == (total_len, feat_cnt)
    assert len(loader) == total_len

    train_len = int(0.8 * (len(temporal_data)))
    assert len(loader.train().ids()) == train_len
    assert len(loader.test().ids()) == len(temporal_data) - train_len
    assert len(loader.sample(100).ids()) == 100

    assert loader.hash() != ""


def test_time_series_survival_dataloader_info() -> None:
    static_data, temporal_data, temporal_horizons, outcome = PBCDataloader().load()
    T, E = outcome

    loader = TimeSeriesSurvivalDataLoader(
        temporal_data=temporal_data,
        temporal_horizons=temporal_horizons,
        static_data=static_data,
        T=T,
        E=E,
        train_size=0.8,
        sensitive_features=["test"],
    )

    max_window_len = max([len(t) for t in temporal_data])

    temporal_features = []
    for item in temporal_data:
        temporal_features.extend(item.columns)
    temporal_features = sorted(np.unique(temporal_features).tolist())

    total_len = 0
    for item in temporal_data:
        total_len += len(item)

    info = loader.info()
    assert info["data_type"] == "time_series_survival"
    assert info["len"] == total_len
    assert info["static_features"] == (
        list(static_data.columns) if static_data is not None else []
    )
    assert info["temporal_features"] == temporal_features
    assert info["outcome_features"] == ["time_to_event", "event"]
    assert info["window_len"] == max_window_len
    assert info["sensitive_features"] == ["test"]

    new_loader = TimeSeriesSurvivalDataLoader.from_info(
        loader.dataframe(), loader.info()
    )
    assert new_loader.shape == loader.shape
    assert new_loader.info() == loader.info()


def test_time_series_survival_create_from_info() -> None:
    static_data, temporal_data, temporal_horizons, outcome = PBCDataloader().load()
    T, E = outcome

    loader = TimeSeriesSurvivalDataLoader(
        temporal_data=temporal_data,
        temporal_horizons=temporal_horizons,
        static_data=static_data,
        T=T,
        E=E,
    )

    df = loader.dataframe()
    reloaded = create_from_info(df, loader.info())

    for col in df.columns:
        assert np.allclose(df[col], reloaded.dataframe()[col], equal_nan=True)


def test_time_series_survival_pack_unpack() -> None:
    static_data, temporal_data, temporal_horizons, outcome = PBCDataloader().load()
    T, E = outcome

    loader = TimeSeriesSurvivalDataLoader(
        temporal_data=temporal_data,
        temporal_horizons=temporal_horizons,
        static_data=static_data,
        T=T,
        E=E,
    )

    _, unp_temporal, unp_temporal_horizons, _, _ = loader.unpack()
    for idx, item in enumerate(unp_temporal):
        assert len(unp_temporal[idx]) == len(temporal_data[idx])
        assert len(unp_temporal_horizons[idx]) == len(temporal_data[idx])

    (
        unp_static_data,
        unp_temporal,
        unp_temporal_horizons,
        unp_out,
    ) = TimeSeriesSurvivalDataLoader.unpack_raw_data(
        loader.dataframe(),
        loader.info(),
    )

    temporal_features = []
    for item in temporal_data:
        temporal_features.extend(item.columns)
    temporal_features = sorted(np.unique(temporal_features).tolist())

    if static_data is not None:
        assert unp_static_data.shape == static_data.shape
        assert (unp_static_data.values == static_data.values).all()
        assert (unp_static_data.columns == static_data.columns).all()
    else:
        assert unp_static_data is None

    assert unp_out.shape == (len(T), 2)
    assert (unp_out["time_to_event"].values == T.values).all()
    assert (unp_out["event"].values == E.values).all()
    assert (unp_out.columns == ["time_to_event", "event"]).all()

    assert len(unp_temporal) == len(temporal_data)
    for idx, item in enumerate(temporal_data):
        assert unp_temporal[idx].shape == (len(item), len(item.columns))
        assert len(unp_temporal_horizons[idx]) == len(item)
        assert sorted(unp_temporal[idx].columns) == sorted(item.columns)


def test_time_series_survival_pack_unpack_numpy() -> None:
    static_data, temporal_data, temporal_horizons, outcome = PBCDataloader().load()
    T, E = outcome

    loader = TimeSeriesSurvivalDataLoader(
        temporal_data=temporal_data,
        temporal_horizons=temporal_horizons,
        static_data=static_data,
        T=T,
        E=E,
    )

    unp_static, unp_temporal, unp_temporal_horizons, unp_T, unp_E = loader.unpack(
        as_numpy=True
    )
    for idx, item in enumerate(unp_temporal):
        assert len(unp_temporal[idx]) == len(temporal_data[idx])
        assert len(unp_temporal_horizons[idx]) == len(temporal_data[idx])

    assert isinstance(unp_static, np.ndarray)
    assert unp_static.shape == static_data.shape
    assert len(unp_T) == len(T)
    assert len(unp_E) == len(E)


@pytest.mark.parametrize("as_numpy", [True, False])
def test_time_series_survival_pack_unpack_padding(as_numpy: bool) -> None:
    static_data, temporal_data, temporal_horizons, outcome = PBCDataloader().load()
    T, E = outcome

    loader = TimeSeriesSurvivalDataLoader(
        temporal_data=temporal_data,
        temporal_horizons=temporal_horizons,
        static_data=static_data,
        T=T,
        E=E,
    )

    max_window_len = max([len(t) for t in temporal_data])
    temporal_features = TimeSeriesDataLoader.unique_temporal_features(temporal_data)

    unp_static, unp_temporal, unp_temporal_horizons, unp_T, unp_E = loader.unpack(
        pad=True,
        as_numpy=as_numpy,
    )
    assert np.asarray(unp_temporal).shape == (
        len(temporal_data),
        max_window_len,
        2 * len(temporal_features),
    )
    assert len(unp_temporal) == len(temporal_data)
    assert unp_temporal[0].shape == (max_window_len, 2 * len(temporal_features))
    assert len(unp_temporal_horizons) == len(temporal_data)
    assert len(unp_temporal_horizons[0]) == max_window_len

    for idx, item in enumerate(unp_temporal):
        assert len(unp_temporal[idx]) == max_window_len
        assert len(unp_temporal_horizons[idx]) == max_window_len
