# synthcity absolute
from synthcity.utils.datasets.time_series.pbc import PBCDataloader


def test_dataloader() -> None:
    loader = PBCDataloader()

    _, temporal_data, outcome = loader.load()

    assert outcome.shape == (len(temporal_data), 2)
    assert len(temporal_data) == 312
    for item in temporal_data:
        assert item.shape[1] == 25
