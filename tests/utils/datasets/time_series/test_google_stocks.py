# synthcity absolute
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader


def test_dataloader() -> None:
    loader = GoogleStocksDataloader(seq_len=20)

    _, temporal_data, observation_times, outcome = loader.load()

    assert outcome.shape == (len(temporal_data), 1)
    assert len(temporal_data) == 40
    assert len(observation_times) == 40
    for idx, item in enumerate(temporal_data):
        assert item.shape == (20, 5)
        assert len(observation_times[idx]) == 20
