# third party
import pytest

# synthcity absolute
from synthcity.utils.datasets.time_series.pbc import PBCDataloader


@pytest.mark.parametrize("as_numpy", [True, False])
def test_dataloader(as_numpy: bool) -> None:
    loader = PBCDataloader(as_numpy=as_numpy)

    static_data, temporal_data, outcome = loader.load()
    t, e, t_ext, e_ext = outcome

    assert len(temporal_data) == 312
    assert static_data.shape == (len(temporal_data), 2)
    assert t.shape == (len(temporal_data),)
    assert e.shape == (len(temporal_data),)
    assert t_ext.shape == (len(temporal_data),)
    assert e_ext.shape == (len(temporal_data),)

    for idx, item in enumerate(temporal_data):
        assert item.shape[1] == 23
        assert item.shape[0] == len(e_ext[idx])
        assert item.shape[0] == len(t_ext[idx])
