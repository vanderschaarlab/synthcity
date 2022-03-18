# third party
import pandas as pd
import pytest

# synthcity absolute
from synthcity.plugins.core.constraints import Constraints


def test_constraint_invalid() -> None:
    with pytest.raises(ValueError):
        Constraints([("fail1", "test")])
    with pytest.raises(ValueError):
        Constraints([("fail1", 0)])
    with pytest.raises(ValueError):
        Constraints([("fail1", "badop", 1)])


def test_constraint_ok() -> None:
    cons = Constraints([("feat1", "lt", 1), ("feat2", "eq", 2)])

    data = pd.DataFrame([[1, 1], [2, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 0
    assert cons.filter(data).sum() == 0

    data = pd.DataFrame([[1, 1], [0, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 1
    assert cons.filter(data).sum() == 1

    data = pd.DataFrame([[-1, 2], [0, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 2
    assert cons.filter(data).sum() == 2


def test_constraint_extend() -> None:
    cons1 = Constraints([("feat1", "le", 1)])
    cons2 = Constraints([("feat2", "eq", 2)])

    data = pd.DataFrame([[1, 1], [0, 2]], columns=["feat1", "feat2"])
    assert len(cons1.match(data)) == 2
    assert cons1.filter(data).sum() == 2

    cons1.extend(cons2)
    assert len(cons1.match(data)) == 1
    assert cons1.filter(data).sum() == 1


def test_constraint_op_lt() -> None:
    cons = Constraints([("feat1", "lt", 1)])

    data = pd.DataFrame([[1, 1], [2, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 0
    assert cons.filter(data).sum() == 0

    data = pd.DataFrame([[1, 1], [0, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 1
    assert cons.filter(data).sum() == 1

    data = pd.DataFrame([[-1, 2], [0, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 2
    assert cons.filter(data).sum() == 2


def test_constraint_op_le() -> None:
    cons = Constraints([("feat1", "le", 1)])

    data = pd.DataFrame([[3, 1], [2, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 0
    assert cons.filter(data).sum() == 0

    data = pd.DataFrame([[1, 1], [2, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 1
    assert cons.filter(data).sum() == 1

    data = pd.DataFrame([[-1, 2], [0, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 2
    assert cons.filter(data).sum() == 2


def test_constraint_op_gt() -> None:
    cons = Constraints([("feat1", "gt", 1)])

    data = pd.DataFrame([[-3, 1], [-2, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 0
    assert cons.filter(data).sum() == 0

    data = pd.DataFrame([[1, 1], [2, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 1
    assert cons.filter(data).sum() == 1

    data = pd.DataFrame([[2, 2], [3, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 2
    assert cons.filter(data).sum() == 2


def test_constraint_op_ge() -> None:
    cons = Constraints([("feat1", "ge", 1)])

    data = pd.DataFrame([[-3, 1], [-2, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 0
    assert cons.filter(data).sum() == 0

    data = pd.DataFrame([[1, 1], [0, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 1
    assert cons.filter(data).sum() == 1

    data = pd.DataFrame([[2, 2], [3, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 2
    assert cons.filter(data).sum() == 2


def test_constraint_op_eq() -> None:
    cons = Constraints([("feat1", "eq", 1)])

    data = pd.DataFrame([[-3, 1], [-2, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 0
    assert cons.filter(data).sum() == 0

    data = pd.DataFrame([[1, 1], [0, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 1
    assert cons.filter(data).sum() == 1

    data = pd.DataFrame([[1, 2], [1, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 2
    assert cons.filter(data).sum() == 2


def test_constraint_op_in() -> None:
    cons = Constraints([("feat1", "in", [1])])

    data = pd.DataFrame([[-3, 1], [-2, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 0
    assert cons.filter(data).sum() == 0

    data = pd.DataFrame([[1, 1], [0, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 1
    assert cons.filter(data).sum() == 1

    data = pd.DataFrame([[1, 2], [1, 2]], columns=["feat1", "feat2"])
    assert len(cons.match(data)) == 2
    assert cons.filter(data).sum() == 2
