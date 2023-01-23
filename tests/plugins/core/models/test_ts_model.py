# stdlib
from typing import Any

# third party
import numpy as np
import pytest

# synthcity absolute
from synthcity.plugins.core.models.ts_model import TimeSeriesModel, modes
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
from synthcity.utils.datasets.time_series.pbc import PBCDataloader
from synthcity.utils.datasets.time_series.sine import SineDataloader


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_rnn_sanity(mode: str, task_type: str) -> None:
    model = TimeSeriesModel(
        task_type=task_type,
        n_static_units_in=3,
        n_temporal_units_in=4,
        n_temporal_window=2,
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
@pytest.mark.parametrize("use_horizon_condition", [True, False])
def test_rnn_regression_fit_predict(
    mode: str, source: Any, use_horizon_condition: bool
) -> None:
    static, temporal, observation_times, outcome = source(as_numpy=True).load()
    outcome = outcome.reshape(-1, 1)

    outlen = len(outcome.reshape(-1)) / len(outcome)

    model = TimeSeriesModel(
        task_type="regression",
        n_static_units_in=static.shape[-1],
        n_temporal_units_in=temporal.shape[-1],
        n_temporal_window=temporal.shape[1],
        output_shape=outcome.shape[1:],
        n_iter=10,
        nonlin_out=[("tanh", outlen)],
        mode=mode,
        use_horizon_condition=use_horizon_condition,
    )

    model.fit(static, temporal, observation_times, outcome)

    y_pred = model.predict(static, temporal, observation_times)

    assert y_pred.shape == outcome.shape

    assert model.score(static, temporal, observation_times, outcome) < 2


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_rnn_classification_fit_predict(mode: str, source: Any) -> None:
    static, temporal, observation_times, outcome = source(as_numpy=True).load()
    static_fake, temporal_fake = np.random.randn(*static.shape), np.random.randn(
        *temporal.shape
    )

    y = np.asarray([1] * len(static) + [0] * len(static_fake))

    model = TimeSeriesModel(
        task_type="classification",
        n_static_units_in=static.shape[-1],
        n_temporal_units_in=temporal.shape[-1],
        n_temporal_window=temporal.shape[1],
        output_shape=[2],
        n_iter=10,
        mode=mode,
    )

    static_data = np.concatenate([static, static_fake])
    temporal_data = np.concatenate([temporal, temporal_fake])
    observation_times = np.concatenate([observation_times, observation_times])

    model.fit(static_data, temporal_data, observation_times, y)

    y_pred = model.predict(static_data, temporal_data, observation_times)

    assert y_pred.shape == y.shape

    print(mode, model.score(static_data, temporal_data, observation_times, y))
    assert model.score(static_data, temporal_data, observation_times, y) <= 1


@pytest.mark.parametrize("mode", modes)
def test_rnn_irregular_ts(mode: str) -> None:
    static, temporal, observation_times, outcome = PBCDataloader(as_numpy=True).load()
    T, E = outcome
    y = np.concatenate([np.expand_dims(T, axis=1), np.expand_dims(E, axis=1)], axis=1)

    model = TimeSeriesModel(
        task_type="regression",
        n_static_units_in=static.shape[-1],
        n_temporal_units_in=temporal[0].shape[-1],
        n_temporal_window=max(len(tmp) for tmp in temporal),
        output_shape=[2],
        n_iter=10,
        mode=mode,
    )

    model.fit(static, temporal, observation_times, y)

    y_pred = model.predict(static, temporal, observation_times)

    assert y_pred.shape == y.shape
