# third party
import torch

# synthcity absolute
from synthcity.plugins.core.models.transformer import TransformerModel
from synthcity.utils.constants import DEVICE
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader


def test_sanity() -> None:
    _, temporal, _, _ = GoogleStocksDataloader(as_numpy=True).load()

    model = TransformerModel(n_units_in=temporal[0].shape[-1], n_units_hidden=10)
    temporal = torch.from_numpy(temporal).to(DEVICE)
    out = model.forward(temporal)

    assert out.shape == (len(temporal), temporal[0].shape[0], 10)
