# third party
import pytest

# synthcity absolute
from synthcity.plugins.core.models.rnn import RNN
from synthcity.utils.datasets.time_series.sine import SineDataloader


@pytest.mark.parametrize("mode", ["LSTM", "RNN", "GRU"])
def test_rnn_sanity(mode: str) -> None:
    model = RNN(
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
def test_rnn_fit(mode: str) -> None:
    data = SineDataloader(no=10, seq_len=5).load()
    X = data[:, :-1, :]
    y = data[:, -1, :]

    outlen = len(y[0, :].reshape(-1))

    model = RNN(
        n_units_in=X.shape[-1],
        n_units_out=outlen,
        window_size=2,
        n_iter=100,
        mode=mode,
    )

    model.fit(X, y)
