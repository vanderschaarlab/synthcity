# synthcity absolute
from synthcity.plugins.core.params import Categorical, Float, Integer


def test_categorical() -> None:
    param = Categorical("test", [1, 2, 3, 22])

    assert param.get() == ["test", [1, 2, 3, 22]]
    assert param.sample() in [1, 2, 3, 22]


def test_integer() -> None:
    param = Integer("test", 0, 100)

    assert param.get() == ["test", 0, 100, 1]
    assert param.sample() in list(range(0, 100))


def test_float() -> None:
    param = Float("test", 0, 1)

    assert param.get() == ["test", 0, 1]
    assert param.sample() < 1
    assert param.sample() > 0
