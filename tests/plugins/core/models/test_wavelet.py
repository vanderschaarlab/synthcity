# third party
import pytest
import torch

# synthcity absolute
from synthcity.plugins.core.models.wavelet import Wavelet
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader


@pytest.mark.parametrize("wavelet", ["haar", "db4"])
def test_sanity(wavelet: str) -> None:
    _, temporal, _, _ = GoogleStocksDataloader(as_numpy=True).load()

    model = Wavelet(
        n_units_in=temporal[0].shape[-1],
        n_units_window=temporal[0].shape[-2],
        n_units_hidden=10,
        wavelet=wavelet,
    )
    temporal = torch.from_numpy(temporal).float()
    out = model.forward(temporal)

    assert out.shape == (len(temporal), 10, 10)
