# third party
import numpy as np
import pytest

# synthcity absolute
from synthcity.plugins.core.models.ts_rnn import TimeSeriesRNN
from synthcity.utils.datasets.time_series.sine import SineDataloader


@pytest.mark.parametrize("mode", ["LSTM", "RNN", "GRU"])
@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_rnn_sanity(mode: str, task_type: str) -> None:
    model = TimeSeriesRNN(
        task_type=task_type,
        n_units_in=3,
        n_units_out=2,
        n_iter=11,
        n_units_hidden=4,
        n_layers=4,
        mode=mode,
        n_iter_print=12,
        batch_size=123,
        lr=1e-2,
        weight_decay=1e-2,
    )

    assert model.n_iter == 11
    assert model.n_units_in == 3
    assert model.n_units_out == 2
    assert model.n_units_hidden == 4
    assert model.n_layers == 4
    assert model.mode == mode
    assert model.n_iter_print == 12
    assert model.batch_size == 123


@pytest.mark.parametrize("mode", ["LSTM", "RNN", "GRU"])
def test_rnn_regression_fit_predict(mode: str) -> None:
    data = SineDataloader(no=10, seq_len=8).load()
    X = data[:, :-2, :]
    y = data[:, -2:, :]

    outlen = len(y[0, :].reshape(-1))

    model = TimeSeriesRNN(
        task_type="regression",
        n_units_in=X.shape[-1],
        n_units_out=outlen,
        window_size=2,
        n_iter=100,
        mode=mode,
    )

    model.fit(X, y)

    y_pred = model.predict(X)

    assert y_pred.shape == y.shape

    assert model.score(X, y) < 1


@pytest.mark.parametrize("mode", ["LSTM", "RNN", "GRU"])
def test_rnn_classification_fit_predict(mode: str) -> None:
    real_data = SineDataloader(no=10, seq_len=8).load()
    fake_data = np.random.randn(*real_data.shape)

    y = np.asarray([1] * len(real_data) + [0] * len(fake_data))

    model = TimeSeriesRNN(
        task_type="classification",
        n_units_in=real_data.shape[-1],
        n_units_out=2,
        window_size=2,
        n_iter=100,
        mode=mode,
    )

    data = np.concatenate([real_data, fake_data])
    model.fit(data, y)

    y_pred = model.predict(data)

    assert y_pred.shape == y.shape

    assert model.score(data, y) <= 1
