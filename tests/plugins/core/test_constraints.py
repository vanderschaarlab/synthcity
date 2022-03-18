# third party
import pandas as pd
import pytest

# synthcity absolute
from synthcity.plugins.core.constraints import Constraints


def test_constraint_invalid() -> None:
    with pytest.raises(ValueError):
        Constraints("debug", {"fail1": "test"})
    with pytest.raises(ValueError):
        Constraints("debug", {"fail1": (0)})
    with pytest.raises(ValueError):
        Constraints("debug", {"fail1": ("badop", 1)})


def test_constraint_ok() -> None:
    cons = Constraints("debug", {"feat1": ("lt", 1), "feat2": ("eq", 2)})

    data = pd.DataFrame([[1, 1], [2, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 0

    data = pd.DataFrame([[1, 1], [0, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 1

    data = pd.DataFrame([[-1, 2], [0, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 2
