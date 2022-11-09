# stdlib
from typing import Any

# third party
import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_iris

# synthcity absolute
from synthcity.plugins.core.models import (
    BinEncoder,
    TabularEncoder,
    TimeSeriesTabularEncoder,
)
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
from synthcity.utils.datasets.time_series.sine import SineDataloader


@pytest.mark.parametrize("encoder", [TabularEncoder, TimeSeriesTabularEncoder])
def test_encoder_sanity(encoder: Any) -> None:
    net = encoder(max_clusters=4)

    assert net.max_clusters == 4


@pytest.mark.parametrize("max_clusters", [4, 10])
def test_encoder_fit_no_discrete_param(max_clusters: int) -> None:
    X, _ = load_diabetes(return_X_y=True, as_frame=True)
    net = TabularEncoder(max_clusters=max_clusters)

    net.fit(X)

    layout = net.layout()

    for column in layout:
        assert column.column_type in ["discrete", "continuous"]
        if column.column_type == "discrete":
            assert len(X[column.column_name].unique()) < 20
            assert column.output_dimensions == len(X[column.column_name].unique())
        else:
            assert len(X[column.column_name].unique()) > 10
            assert column.output_dimensions <= 1 + max_clusters


@pytest.mark.parametrize("max_clusters", [4, 10])
def test_encoder_fit_discrete_param(max_clusters: int) -> None:
    X, _ = load_diabetes(return_X_y=True, as_frame=True)
    net = TabularEncoder(max_clusters=max_clusters)

    net.fit(X, discrete_columns=["sex"])

    layout = net.layout()

    for column in layout:
        assert column.column_type in ["discrete", "continuous"]
        if column.column_type == "discrete":
            assert len(X[column.column_name].unique()) < 20
            assert column.output_dimensions == len(X[column.column_name].unique())
        else:
            assert len(X[column.column_name].unique()) > 10
            assert column.output_dimensions <= 1 + max_clusters


@pytest.mark.parametrize("max_clusters", [4, 10])
def test_encoder_fit_transform(max_clusters: int) -> None:
    X, _ = load_diabetes(return_X_y=True, as_frame=True)
    net = TabularEncoder(max_clusters=max_clusters)

    encoded = net.fit_transform(X)
    layout = net.layout()

    assert (X.index == encoded.index).all()

    for column in layout:
        if column.column_type == "discrete":
            for val in X[column.column_name].unique():
                assert f"{column.column_name}_{val}" in encoded.columns
                assert set(encoded[f"{column.column_name}_{val}"].unique()) == set(
                    [0, 1]
                )

        else:
            assert f"{column.column_name}.normalized" in encoded.columns
            for enc_col in encoded.columns:
                if column.column_name in enc_col and "normalized" not in enc_col:
                    assert set(encoded[enc_col].unique()) == set([0, 1])


@pytest.mark.parametrize("max_clusters", [20, 50])
def test_encoder_inverse_transform(max_clusters: int) -> None:
    X, _ = load_diabetes(return_X_y=True, as_frame=True)
    net = TabularEncoder(max_clusters=max_clusters)

    net.fit(X)

    encoded = net.transform(X)
    recovered = net.inverse_transform(encoded)

    assert X.shape == recovered.shape
    assert (X.columns == recovered.columns).all()
    assert np.abs(X - recovered).sum().sum() < 5


def test_encoder_activation_layout() -> None:
    X, _ = load_diabetes(return_X_y=True, as_frame=True)
    net = TabularEncoder()

    net.fit(X)
    act_layout = net.activation_layout(
        discrete_activation="softmax", continuous_activation="tanh"
    )
    layout = net.layout()

    assert len(layout) <= len(act_layout)

    act_step = 0

    for col_info in layout:
        if col_info.column_type == "continuous":
            assert act_layout[act_step] == ("tanh", 1)
            assert act_layout[act_step + 1] == (
                "softmax",
                col_info.output_dimensions - 1,
            )
            act_step += 2
        else:
            assert act_layout[act_step] == ("softmax", col_info.output_dimensions)
            act_step += 1


def test_bin_encoder() -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    net = BinEncoder(max_clusters=10)

    net.fit(X)
    binned = net.transform(X)

    for col in X.columns:
        assert len(binned[col].unique()) <= 10


@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_ts_encoder_fit(source: Any) -> None:
    max_clusters = 5
    categorical_limit = 10
    static, temporal, temporal_horizons, _ = source().load()
    net = TimeSeriesTabularEncoder(
        max_clusters=max_clusters, categorical_limit=categorical_limit
    ).fit(static, temporal, temporal_horizons)

    static_layout, temporal_layout = net.layout()

    for column in static_layout:
        assert column.column_type in ["discrete", "continuous"]
        if column.column_type == "discrete":
            assert len(static[column.column_name].unique()) < categorical_limit
            assert column.output_dimensions == len(static[column.column_name].unique())
        else:
            assert len(static[column.column_name].unique()) >= categorical_limit
            assert column.output_dimensions <= 1 + max_clusters

    for column in temporal_layout:
        assert column.column_type in ["discrete", "continuous"]
        if column.column_type == "discrete":
            assert len(temporal[0][column.column_name].unique()) < categorical_limit
            assert column.output_dimensions == len(
                temporal[0][column.column_name].unique()
            )
        else:
            assert len(temporal[0][column.column_name].unique()) > 0
            assert column.output_dimensions <= 1 + max_clusters


@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_ts_encoder_fit_transform(source: Any) -> None:
    max_clusters = 5
    categorical_limit = 10
    static, temporal, temporal_horizons, _ = source().load()
    net = TimeSeriesTabularEncoder(
        max_clusters=max_clusters, categorical_limit=categorical_limit
    ).fit(static, temporal, temporal_horizons)

    static_layout, temporal_layout = net.layout()
    static_encoded, temporal_encoded, horizons_encoded = net.fit_transform(
        static, temporal, temporal_horizons
    )

    assert (static.index == static_encoded.index).all()
    for column in static_layout:
        if column.column_type == "discrete":
            for val in static[column.column_name].unique():
                assert f"{column.column_name}_{val}" in static_encoded.columns
                assert set(
                    static_encoded[f"{column.column_name}_{val}"].unique()
                ) == set([0, 1])

        else:
            assert f"{column.column_name}.normalized" in static_encoded.columns
            for enc_col in static_encoded.columns:
                if column.column_name in enc_col and "normalized" not in enc_col:
                    assert set(static_encoded[enc_col].unique().astype(int)).issubset(
                        set([0, 1])
                    )

    assert len(temporal) == len(temporal_encoded)

    for idx, item in enumerate(temporal_encoded):
        assert len(item) == len(horizons_encoded[idx])
        for column in temporal_layout:
            if column.column_type == "discrete":
                for val in temporal[0][column.column_name].unique():
                    assert f"{column.column_name}_{val}" in item.columns
                    assert set(item[f"{column.column_name}_{val}"].unique()) == set(
                        [0, 1]
                    )

            else:
                assert f"{column.column_name}.normalized" in item.columns
                for enc_col in item.columns:
                    if column.column_name in enc_col and "normalized" not in enc_col:
                        assert set(item[enc_col].unique().astype(int)).issubset(
                            set([0, 1])
                        )


@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_ts_encoder_inverse_transform(source: Any) -> None:
    max_clusters = 5
    categorical_limit = 10
    static, temporal, temporal_horizons, _ = source().load()
    net = TimeSeriesTabularEncoder(
        max_clusters=max_clusters, categorical_limit=categorical_limit
    ).fit(static, temporal, temporal_horizons)

    static_layout, temporal_layout = net.layout()
    static_encoded, temporal_encoded, horizons_encoded = net.fit_transform(
        static, temporal, temporal_horizons
    )
    static_reversed, temporal_reversed, horizons_reversed = net.inverse_transform(
        static_encoded, temporal_encoded, horizons_encoded
    )

    assert (static_reversed.index == static.index).all()
    assert static_reversed.shape == static.shape
    assert (static_reversed.columns == static.columns).all()
    assert (
        np.abs(np.asarray(temporal_horizons) - np.asarray(horizons_reversed))
        .sum()
        .sum()
        < 1
    )
    assert np.abs(static - static_reversed).sum().sum() < 5

    assert len(temporal) == len(temporal_reversed)
    for idx, temporal_decoded in enumerate(temporal_reversed):
        assert temporal_decoded.shape == temporal[idx].shape
        assert (temporal_decoded.columns == temporal[idx].columns).all()
        assert np.abs(temporal_decoded - temporal[idx]).sum().sum() < 5


@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_ts_encoder_activation_layout(source: Any) -> None:
    max_clusters = 5
    categorical_limit = 10
    static, temporal, horizons, _ = source().load()
    net = TimeSeriesTabularEncoder(
        max_clusters=max_clusters, categorical_limit=categorical_limit
    ).fit(static, temporal, horizons)

    static_act_layout, temporal_act_layout = net.activation_layout(
        discrete_activation="softmax", continuous_activation="tanh"
    )
    static_layout, temporal_layout = net.layout()

    assert len(static_layout) <= len(static_act_layout)
    assert len(temporal_layout) <= len(temporal_act_layout)

    act_step = 0
    for col_info in static_layout:
        if col_info.column_type == "continuous":
            assert static_act_layout[act_step] == ("tanh", 1)
            assert static_act_layout[act_step + 1] == (
                "softmax",
                col_info.output_dimensions - 1,
            )
            act_step += 2
        else:
            assert static_act_layout[act_step] == (
                "softmax",
                col_info.output_dimensions,
            )
            act_step += 1

    act_step = 0
    for col_info in temporal_layout:
        if col_info.column_type == "continuous":
            assert temporal_act_layout[act_step] == ("tanh", 1)
            assert temporal_act_layout[act_step + 1] == (
                "softmax",
                col_info.output_dimensions - 1,
            )
            act_step += 2
        else:
            assert temporal_act_layout[act_step] == (
                "softmax",
                col_info.output_dimensions,
            )
            act_step += 1
