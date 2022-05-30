# synthcity absolute
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader


def test_dataloader() -> None:
    loader = GoogleStocksDataloader(seq_len=11)

    dataset = loader.load()

    assert dataset.shape == (87, 11, 7)
