# synthcity absolute
from synthcity.utils.datasets.time_series.sine import SineDataloader


def test_dataloader() -> None:
    loader = SineDataloader(no=5, seq_len=10, dim=15, freq_scale=0.5)

    dataset = loader.load()

    assert dataset.shape == (5, 10, 15)
