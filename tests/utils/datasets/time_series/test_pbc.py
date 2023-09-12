# third party
import pytest

# synthcity absolute
from synthcity.utils.datasets.time_series.pbc import PBCDataloader


@pytest.mark.parametrize("as_numpy", [True, False])
def test_dataloader(as_numpy: bool) -> None:
    loader = PBCDataloader(as_numpy=as_numpy)

    static_data, temporal_data, observation_times, outcome = loader.load()
    t, e = outcome

    assert len(temporal_data) == 312
    assert static_data.shape == (len(temporal_data), 1)
    assert t.shape == (len(temporal_data),)
    assert e.shape == (len(temporal_data),)
    assert len(observation_times) == len(temporal_data)

    for idx, item in enumerate(temporal_data):
        assert item.shape[1] == 14
        assert len(observation_times[idx]) == item.shape[0]
