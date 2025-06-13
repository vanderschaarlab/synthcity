# stdlib
import sys
from datetime import datetime
from typing import Any

# third party
import numpy as np
import pandas as pd
import pytest
import torch
from lifelines.datasets import load_rossi
from sklearn.datasets import load_breast_cancer
from torchvision import datasets, transforms

# synthcity absolute
from synthcity.plugins.core.dataloader import (
    GenericDataLoader,
    ImageDataLoader,
    SurvivalAnalysisDataLoader,
    Syn_SeqDataLoader,
    TimeSeriesDataLoader,
    TimeSeriesSurvivalDataLoader,
    create_from_info,
)
from synthcity.plugins.core.dataset import FlexibleDataset, TensorDataset
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
from synthcity.utils.datasets.time_series.pbc import PBCDataloader
from synthcity.utils.datasets.time_series.sine import SineDataloader


def test_generic_dataloader_sanity() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    X["target"] = y

    loader = GenericDataLoader(X, target_column="target")

    assert loader.is_tabular()
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

    loader = GenericDataLoader(X, target_column="target", train_size=0.5)
    assert abs(len(loader.train()) - len(loader.test())) < 2


def test_generic_dataloader_encoder() -> None:
    def _get_dtypes(df: pd.DataFrame) -> list:
        return list(df.infer_objects().dtypes)

    test = pd.DataFrame(
        [
            [0, 0.344, "cat1", datetime.now()],
            [1, 0.444, "cat1", datetime.now()],
            [0, 0.544, "cat2", datetime.now()],
        ]
    )
    loader = GenericDataLoader(test)
    dtypes = _get_dtypes(test)

    encoded, encoders = loader.encode()
    encoded_dtypes = _get_dtypes(encoded.dataframe())
    for dt in encoded_dtypes:
        assert dt in ["float64", "int64", "float", "int"]

    assert (encoded.columns == test.columns).all()

    decoded = encoded.decode(encoders)
    decoded_dtypes = _get_dtypes(decoded.dataframe())

    assert (decoded.columns == test.columns).all()
    for idx, dt in enumerate(dtypes):
        assert dt == decoded_dtypes[idx]


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

    assert loader.is_tabular()
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
    loader = SurvivalAnalysisDataLoader(
        df,
        time_to_event_column=time_to_event_column,
        target_column=target_column,
        time_horizons=[20],
        train_size=0.5,
    )

    assert abs(len(loader.train()) - len(loader.test())) < 2


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
    static_data, temporal_data, observation_times, outcome = source().load()

    loader = TimeSeriesDataLoader(
        temporal_data=temporal_data,
        observation_times=observation_times,
        static_data=static_data,
        outcome=outcome,
        train_size=0.8,
    )

    assert len(loader.raw()) == 5
    assert loader.is_tabular()

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
    static_data, temporal_data, observation_times, outcome = source().load()

    loader = TimeSeriesDataLoader(
        temporal_data=temporal_data,
        observation_times=observation_times,
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
    static_data, temporal_data, observation_times, outcome = source.load()

    loader = TimeSeriesDataLoader(
        temporal_data=temporal_data,
        observation_times=observation_times,
        static_data=static_data,
        outcome=outcome,
    )

    if repack:
        (
            unp_static_data,
            unp_temporal,
            unp_observation_times,
            unp_out,
        ) = TimeSeriesDataLoader.unpack_raw_data(
            loader.dataframe(),
            loader.info(),
        )
    else:
        unp_static_data, unp_temporal, unp_observation_times, unp_out = loader.unpack()

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
        assert len(unp_observation_times[idx]) == len(observation_times[idx])
        assert sorted(unp_temporal[idx].columns) == sorted(item.columns)
        cols = list(unp_temporal[idx].columns)
        assert (unp_temporal[idx].values == item[cols].values).all()


def test_time_series_survival_dataloader_sanity() -> None:
    static_data, temporal_data, observation_times, outcome = PBCDataloader().load()
    T, E = outcome

    loader = TimeSeriesSurvivalDataLoader(
        temporal_data=temporal_data,
        observation_times=observation_times,
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
    static_data, temporal_data, observation_times, outcome = PBCDataloader().load()
    T, E = outcome

    loader = TimeSeriesSurvivalDataLoader(
        temporal_data=temporal_data,
        observation_times=observation_times,
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
    static_data, temporal_data, observation_times, outcome = PBCDataloader().load()
    T, E = outcome

    loader = TimeSeriesSurvivalDataLoader(
        temporal_data=temporal_data,
        observation_times=observation_times,
        static_data=static_data,
        T=T,
        E=E,
    )

    df = loader.dataframe()
    reloaded = create_from_info(df, loader.info())

    for col in df.columns:
        assert np.allclose(df[col], reloaded.dataframe()[col], equal_nan=True)


def test_time_series_survival_pack_unpack() -> None:
    static_data, temporal_data, observation_times, outcome = PBCDataloader().load()
    T, E = outcome

    loader = TimeSeriesSurvivalDataLoader(
        temporal_data=temporal_data,
        observation_times=observation_times,
        static_data=static_data,
        T=T,
        E=E,
    )

    _, unp_temporal, unp_observation_times, _, _ = loader.unpack()
    for idx, item in enumerate(unp_temporal):
        assert len(unp_temporal[idx]) == len(temporal_data[idx])
        assert len(unp_observation_times[idx]) == len(temporal_data[idx])

    (
        unp_static_data,
        unp_temporal,
        unp_observation_times,
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
        assert len(unp_observation_times[idx]) == len(item)
        assert sorted(unp_temporal[idx].columns) == sorted(item.columns)


def test_time_series_survival_pack_unpack_numpy() -> None:
    static_data, temporal_data, observation_times, outcome = PBCDataloader().load()
    T, E = outcome

    loader = TimeSeriesSurvivalDataLoader(
        temporal_data=temporal_data,
        observation_times=observation_times,
        static_data=static_data,
        T=T,
        E=E,
    )

    unp_static, unp_temporal, unp_observation_times, unp_T, unp_E = loader.unpack(
        as_numpy=True
    )
    for idx, item in enumerate(unp_temporal):
        assert len(unp_temporal[idx]) == len(temporal_data[idx])
        assert len(unp_observation_times[idx]) == len(temporal_data[idx])

    assert isinstance(unp_static, np.ndarray)
    assert unp_static.shape == static_data.shape
    assert len(unp_T) == len(T)
    assert len(unp_E) == len(E)


@pytest.mark.parametrize("as_numpy", [True, False])
def test_time_series_survival_pack_unpack_padding(as_numpy: bool) -> None:
    static_data, temporal_data, observation_times, outcome = PBCDataloader().load()
    T, E = outcome

    loader = TimeSeriesSurvivalDataLoader(
        temporal_data=temporal_data,
        observation_times=observation_times,
        static_data=static_data,
        T=T,
        E=E,
    )

    max_window_len = max([len(t) for t in temporal_data])
    temporal_features = TimeSeriesDataLoader.unique_temporal_features(temporal_data)

    unp_static, unp_temporal, unp_observation_times, unp_T, unp_E = loader.unpack(
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
    assert len(unp_observation_times) == len(temporal_data)
    assert len(unp_observation_times[0]) == max_window_len

    for idx, item in enumerate(unp_temporal):
        assert len(unp_temporal[idx]) == max_window_len
        assert len(unp_observation_times[idx]) == max_window_len


@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
@pytest.mark.parametrize("height", [55, 64])
@pytest.mark.parametrize("width", [32, 22])
def test_image_dataloader_sanity(height: int, width: int) -> None:
    dataset = datasets.MNIST(".", download=True)

    loader = ImageDataLoader(
        data=dataset,
        train_size=0.8,
        height=height,
        width=width,
    )
    channels = 1

    assert loader.shape == (len(dataset), channels, height, width)
    assert loader.info()["height"] == height
    assert loader.info()["width"] == width
    assert loader.info()["channels"] == channels
    assert loader.info()["len"] == len(dataset)
    assert not loader.is_tabular()

    assert isinstance(loader.unpack(), torch.utils.data.Dataset)

    assert loader.sample(5).shape == (5, channels, height, width)

    assert loader[0].shape == (channels, height, width)

    assert loader.hash() != ""

    assert loader.train().shape == (0.8 * len(dataset), channels, height, width)
    assert loader.test().shape == (0.2 * len(dataset), channels, height, width)

    x_np = loader.numpy()
    assert x_np.shape == (len(dataset), channels, height, width)
    assert isinstance(x_np, np.ndarray)

    df = loader.dataframe()
    assert df.shape == (len(dataset), channels * height * width)
    assert isinstance(df, pd.DataFrame)

    assert loader.unpack().labels().shape == (len(loader),)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
def test_image_dataloader_create_from_info() -> None:
    dataset = datasets.MNIST(".", download=True)

    loader = ImageDataLoader(
        data=dataset,
        train_size=0.8,
        height=32,
    )

    data = loader.unpack()

    reloaded = create_from_info(data, loader.info())

    for key in loader.info():
        assert reloaded.info()[key] == loader.info()[key]


def test_image_dataloader_create_from_tensor() -> None:
    X = torch.randn((100, 10, 10))
    y = torch.randn((100,))

    loader = ImageDataLoader(
        data=(X, y),
        train_size=0.8,
        height=32,
    )

    assert len(loader) == len(X)
    assert loader.shape == (100, 1, 32, 32)


def test_image_datasets() -> None:
    size = 100
    X = torch.rand(size, 10, 10)
    y = torch.rand(size)

    gen_dataset = TensorDataset(images=X, targets=y)
    assert (gen_dataset[0][0] == X[0]).all()
    assert (gen_dataset[0][1] == y[0]).all()

    img_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((20, 20)),
            transforms.ToTensor(),
        ]
    )

    transform_dataset = FlexibleDataset(gen_dataset, transform=img_transform)
    assert transform_dataset.shape() == (size, 1, 20, 20)
    assert transform_dataset[0][0].shape == (1, 20, 20)
    assert transform_dataset[0][1] == y[0]

    gen_dataset = TensorDataset(images=X, targets=None)
    assert (gen_dataset[0][0] == X[0]).all()
    assert gen_dataset[0][1] is None

    transform_dataset = FlexibleDataset(gen_dataset, transform=img_transform)
    assert transform_dataset.shape() == (size, 1, 20, 20)
    assert transform_dataset[0][0].shape == (1, 20, 20)
    assert transform_dataset[0][1] is None

    transform_dataset = transform_dataset.filter_indices([0, 1, 2])
    assert len(transform_dataset) == 3
    assert (transform_dataset.indices == [0, 1, 2]).all()

    transform_dataset = transform_dataset.filter_indices([1, 2])
    assert len(transform_dataset) == 2
    assert (transform_dataset.indices == [1, 2]).all()

    transform_dataset = transform_dataset.filter_indices([0])
    assert len(transform_dataset) == 1
    assert (transform_dataset.indices == [1]).all()


def test_syn_seq_dataloader_sanity() -> None:
    """
    Test basic creation of Syn_SeqDataLoader, verifying:
      - shape
      - cat columns are auto-injected
      - train/test split
      - info() returns correct encoder details
    """
    df = pd.DataFrame(
        {
            "income": [100, 200, -8, 400],
            "income_cat": ["-777", "-777", "-8", "-777"],
            "colX": [1, 2, 3, 4],
        }
    )
    user_custom = {
        "syn_order": ["income", "colX"],
        "method": {"colX": "rf"},
        "variable_selection": {"colX": ["income"]},
    }
    loader = Syn_SeqDataLoader(
        data=df,
        user_custom=user_custom,
        target_column="income",
        random_state=42,
        train_size=0.5,
        verbose=False,
    )

    assert loader.shape == (4, 3)
    info = loader.info()
    syn_order = info.get("syn_order", [])
    assert "income_cat" in syn_order, "Should auto-inject 'income_cat' into syn_order"

    tr_loader = loader.train()
    te_loader = loader.test()
    assert len(tr_loader) + len(te_loader) == 4


def test_syn_seq_dataloader_sample_and_drop() -> None:
    """
    Test sample() and drop() with the Syn_SeqDataLoader.
    """
    df = pd.DataFrame(
        {
            "colA": [1, 2, 3, 4, 5],
            "colB_cat": ["-777", "-777", "-8", "-777", "-8"],
            "colC": [10, 20, 30, 40, 50],
        }
    )
    loader = Syn_SeqDataLoader(df)
    samp = loader.sample(2)
    assert len(samp) == 2

    dropped = loader.drop(columns=["colC"])
    assert "colC" not in dropped.columns
    assert dropped.shape == (5, 2)


def test_syn_seq_dataloader_encode_decode() -> None:
    """
    Check that parent's encode()/decode() (label encoding logic) works
    with Syn_SeqDataLoader.
    """
    df = pd.DataFrame({"col1": ["A", "B", "A", "C"], "col2": [10, 20, 30, 40]})
    loader = Syn_SeqDataLoader(data=df, user_custom={})

    encoded_loader, encoders = loader.encode()
    assert encoded_loader.shape == df.shape

    decoded_loader = encoded_loader.decode(encoders)
    df_dec = decoded_loader.dataframe()
    assert (df_dec["col1"].values == df["col1"].values).all()
    assert (df_dec["col2"].values == df["col2"].values).all()


def test_syn_seq_dataloader_info() -> None:
    """
    Check that loader.info() merges base_info and encoder info correctly.
    """
    df = pd.DataFrame(
        {"colX": ["x", "y", "z"], "colY": [1, 2, 3], "colY_cat": ["-777", "-777", "-8"]}
    )
    user_custom = {
        "syn_order": ["colX", "colY"],
        "method": {"colY": "rf"},
        "variable_selection": {},
    }
    loader = Syn_SeqDataLoader(data=df, user_custom=user_custom, verbose=False)
    info = loader.info()

    print("info['method']:", info["method"])
    print("info['syn_order']:", info["syn_order"])
    sys.stdout.flush()

    assert info["data_type"] == "syn_seq"
    assert info["len"] == 3

    assert "syn_order" in info, "encoder's syn_order should be in info"
    assert "variable_selection" in info, "encoder's varsel should be in info"
    assert "colY_cat" in info["syn_order"]
    assert info["method"].get("colY") == "rf"


def test_syn_seq_dataloader_pack_unpack() -> None:
    X = pd.DataFrame({"colX": ["x", "y", "z"], "colY_cat": ["-777", "-777", "-8"]})
    y = pd.Series([1, 2, 3], name="colY")
    X["colY"] = y

    user_custom = {
        "syn_order": ["colX", "colY"],
        "method": {"colY": "rf"},
        "variable_selection": {},
    }
    loader = Syn_SeqDataLoader(
        data=X, user_custom=user_custom, target_column="colY", verbose=False
    )

    Xu, yu = loader.unpack()
    assert Xu.shape == (len(X), X.shape[1] - 1)
    assert yu.shape == y.shape


def test_syn_seq_dataloader_compression() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X["domain"] = y

    user_custom = {
        "syn_order": ["colX", "colY"],
        "method": {"colY": "rf"},
        "variable_selection": {},
    }
    loader = Syn_SeqDataLoader(X, user_custom=user_custom)

    compressed, context = loader.compress()

    assert len(compressed) == len(loader)
    assert compressed.shape[1] <= loader.shape[1]
    assert "domain" in compressed.columns

    decompressed = compressed.decompress(context)

    assert len(decompressed) == len(loader)
    assert decompressed.shape[1] == loader.shape[1]
