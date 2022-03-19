# synthcity absolute
from synthcity.plugins.core.params import Categorical, Float, Integer


def test_categorical() -> None:
    param = Categorical(name="test", choices=[1, 2, 3, 22])

    assert param.get() == ["test", [1, 2, 3, 22]]
    assert param.sample() in [1, 2, 3, 22]
    assert param.has(1)
    assert not param.has(5)
    assert len(param.as_constraint().rules) == 1

    param_other = Categorical(name="test", choices=[1, 2])
    assert param.includes(param_other)
    assert not param_other.includes(param)

    param_other = Categorical(name="test", choices=[1, 2, 555])
    assert not param.includes(param_other)
    assert not param_other.includes(param)

    param_other = Categorical(name="test", choices=[1, 2, 3, 4, 22])
    assert not param.includes(param_other)
    assert param_other.includes(param)

    param_other = Categorical(name="test", choices=[1, 2, 3, 22])
    assert param.includes(param_other)
    assert param_other.includes(param)


def test_integer() -> None:
    param = Integer(name="test", low=0, high=100)

    assert param.get() == ["test", 0, 100, 1]
    assert param.sample() in list(range(0, 101))
    assert param.has(1)
    assert not param.has(101)
    assert not param.has(-1)
    assert len(param.as_constraint().rules) == 2

    param_other = Integer(name="test", low=1, high=99)
    assert param.includes(param_other)
    assert not param_other.includes(param)

    param_other = Integer(name="test", low=1, high=101)
    assert not param.includes(param_other)
    assert not param_other.includes(param)

    param_other = Integer(name="test", low=-1, high=10)
    assert not param.includes(param_other)
    assert not param_other.includes(param)

    param_other = Integer(name="test", low=-1, high=1000)
    assert not param.includes(param_other)
    assert param_other.includes(param)

    param_other = Integer(name="test", low=0, high=100)
    assert param.includes(param_other)
    assert param_other.includes(param)


def test_float() -> None:
    param = Float(name="test", low=0, high=1)

    assert param.get() == ["test", 0, 1]
    assert param.sample() < 1
    assert param.sample() > 0
    assert len(param.as_constraint().rules) == 2

    param_other = Float(name="test", low=0, high=0.5)
    assert param.includes(param_other)
    assert not param_other.includes(param)

    param_other = Float(name="test", low=0.1, high=1.1)
    assert not param.includes(param_other)
    assert not param_other.includes(param)

    param_other = Float(name="test", low=-1, high=0.5)
    assert not param.includes(param_other)
    assert not param_other.includes(param)

    param_other = Float(name="test", low=-1, high=1000)
    assert not param.includes(param_other)
    assert param_other.includes(param)

    param_other = Float(name="test", low=0, high=1)
    assert param.includes(param_other)
    assert param_other.includes(param)
    assert param.has(1)
    assert not param.has(2)
    assert not param.has(-1)
