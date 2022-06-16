# synthcity absolute
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader


def test_dataloader() -> None:
    loader = GoogleStocksDataloader(seq_len=20)

    _, temporal_data, outcome = loader.load()

    assert outcome.shape == (len(temporal_data), 1)
    assert len(temporal_data) == 40
    for item in temporal_data:
        assert item.shape == (20, 5)
