# stdlib
from typing import Any

# third party
import numpy as np
import pytest

# synthcity absolute
from synthcity.plugins.core.models.ts_model import TimeSeriesModel
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
from synthcity.utils.datasets.time_series.sine import SineDataloader

modes = [
    "LSTM",
    "GRU",
    "RNN",
    "MLSTM_FCN",
    "TCN",
    "InceptionTime",
    "InceptionTimePlus",
    "XceptionTime",
    "ResCNN",
    "OmniScaleCNN",
    "TST",
    "XCM",
    "gMLP",
    "MiniRocket",
    "MiniRocketPlus",
    "TransformerModel",
    "TSiTPlus",
    "TSTPlus",
    "TSPerceiver",
]


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_rnn_sanity(mode: str, task_type: str) -> None:
    model = TimeSeriesModel(
        task_type=task_type,
        n_static_units_in=3,
        n_temporal_units_in=4,
        output_shape=[2],
        n_iter=11,
        n_static_units_hidden=41,
        n_temporal_units_hidden=42,
        n_static_layers_hidden=2,
        n_temporal_layers_hidden=3,
        mode=mode,
        n_iter_print=12,
        batch_size=123,
        lr=1e-2,
        weight_decay=1e-2,
    )

    assert model.n_iter == 11
    assert model.n_static_units_in == 3
    assert model.n_temporal_units_in == 4
    assert model.n_units_out == 2
    assert model.output_shape == [2]
    assert model.n_static_units_hidden == 41
    assert model.n_temporal_units_hidden == 42
    assert model.n_static_layers_hidden == 2
    assert model.n_temporal_layers_hidden == 3
    assert model.mode == mode
    assert model.n_iter_print == 12
    assert model.batch_size == 123


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_rnn_regression_fit_predict(mode: str, source: Any) -> None:
    static, temporal, temporal_horizons, outcome = source(as_numpy=True).load()
    outcome = outcome.reshape(-1, 1)

    outlen = len(outcome.reshape(-1)) / len(outcome)

    model = TimeSeriesModel(
        task_type="regression",
        n_static_units_in=static.shape[-1],
        n_temporal_units_in=temporal.shape[-1],
        output_shape=outcome.shape[1:],
        window_size=2,
        n_iter=10,
        nonlin_out=[("tanh", outlen)],
        mode=mode,
    )

    model.fit(static, temporal, temporal_horizons, outcome)

    y_pred = model.predict(static, temporal, temporal_horizons)

    assert y_pred.shape == outcome.shape

    assert model.score(static, temporal, temporal_horizons, outcome) < 2


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_rnn_classification_fit_predict(mode: str, source: Any) -> None:
    static, temporal, temporal_horizons, outcome = source(as_numpy=True).load()
    static_fake, temporal_fake = np.random.randn(*static.shape), np.random.randn(
        *temporal.shape
    )

    y = np.asarray([1] * len(static) + [0] * len(static_fake))

    model = TimeSeriesModel(
        task_type="classification",
        n_static_units_in=static.shape[-1],
        n_temporal_units_in=temporal.shape[-1],
        output_shape=[2],
        window_size=2,
        n_iter=10,
        mode=mode,
    )

    static_data = np.concatenate([static, static_fake])
    temporal_data = np.concatenate([temporal, temporal_fake])
    temporal_horizons = np.concatenate([temporal_horizons, temporal_horizons])

    model.fit(static_data, temporal_data, temporal_horizons, y)

    y_pred = model.predict(static_data, temporal_data, temporal_horizons)

    assert y_pred.shape == y.shape

    print(mode, model.score(static_data, temporal_data, temporal_horizons, y))
    assert model.score(static_data, temporal_data, temporal_horizons, y) <= 1
