# synthcity absolute
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    FloatDistribution,
    IntegerDistribution,
    constraint_to_distribution,
)


def test_categorical() -> None:
    param = CategoricalDistribution(name="test", choices=[1, 2, 3, 22])

    assert param.get() == ["test", [1, 2, 3, 22]]
    assert len(param.sample(count=5)) == 5
    for sample in param.sample(count=5):
        assert sample in [1, 2, 3, 22]

    assert param.has(1)
    assert not param.has(5)
    assert len(param.as_constraint().rules) == 1

    param_other = CategoricalDistribution(name="test", choices=[1, 2])
    assert param.includes(param_other)
    assert not param_other.includes(param)

    param_other = CategoricalDistribution(name="test", choices=[1, 2, 555])
    assert not param.includes(param_other)
    assert not param_other.includes(param)

    param_other = CategoricalDistribution(name="test", choices=[1, 2, 3, 4, 22])
    assert not param.includes(param_other)
    assert param_other.includes(param)

    param_other = CategoricalDistribution(name="test", choices=[1, 2, 3, 22])
    assert param.includes(param_other)
    assert param_other.includes(param)


def test_integer() -> None:
    param = IntegerDistribution(name="test", low=0, high=100)

    assert param.get() == ["test", 0, 100, 1]
    assert len(param.sample(count=5)) == 5
    for sample in param.sample(count=5):
        assert sample in list(range(0, 101))
    assert param.has(1)
    assert not param.has(101)
    assert not param.has(-1)
    assert len(param.as_constraint().rules) == 3

    param_other = IntegerDistribution(name="test", low=1, high=99)
    assert param.includes(param_other)
    assert not param_other.includes(param)

    param_other = IntegerDistribution(name="test", low=1, high=101)
    assert not param.includes(param_other)
    assert not param_other.includes(param)

    param_other = IntegerDistribution(name="test", low=-1, high=10)
    assert not param.includes(param_other)
    assert not param_other.includes(param)

    param_other = IntegerDistribution(name="test", low=-1, high=1000)
    assert not param.includes(param_other)
    assert param_other.includes(param)

    param_other = IntegerDistribution(name="test", low=0, high=100)
    assert param.includes(param_other)
    assert param_other.includes(param)


def test_float() -> None:
    param = FloatDistribution(name="test", low=0, high=1)

    assert param.get() == ["test", 0, 1]
    assert len(param.sample(count=5)) == 5
    for sample in param.sample(count=5):
        assert sample < 1
        assert sample > 0
    assert len(param.as_constraint().rules) == 3

    param_other = FloatDistribution(name="test", low=0, high=0.5)
    assert param.includes(param_other)
    assert not param_other.includes(param)

    param_other = FloatDistribution(name="test", low=0.1, high=1.1)
    assert not param.includes(param_other)
    assert not param_other.includes(param)

    param_other = FloatDistribution(name="test", low=-1, high=0.5)
    assert not param.includes(param_other)
    assert not param_other.includes(param)

    param_other = FloatDistribution(name="test", low=-1, high=1000)
    assert not param.includes(param_other)
    assert param_other.includes(param)

    param_other = FloatDistribution(name="test", low=0, high=1)
    assert param.includes(param_other)
    assert param_other.includes(param)
    assert param.has(1)
    assert not param.has(2)
    assert not param.has(-1)


def test_categorical_constraint_to_distribution() -> None:
    param = CategoricalDistribution(name="test", choices=[1, 2, 3, 22])
    constraint = param.as_constraint()
    new_dist = constraint_to_distribution(constraint, "test")

    assert isinstance(new_dist, CategoricalDistribution)
    assert new_dist.name == "test"
    assert new_dist.choices == [1, 2, 3, 22]


def test_int_constraint_to_distribution() -> None:
    param = IntegerDistribution(name="test", low=-1, high=5)
    constraint = param.as_constraint()
    new_dist = constraint_to_distribution(constraint, "test")

    assert isinstance(new_dist, IntegerDistribution)
    assert new_dist.name == "test"
    assert new_dist.low == -1
    assert new_dist.high == 5


def test_float_constraint_to_distribution() -> None:
    param = FloatDistribution(name="test", low=-1, high=5)
    constraint = param.as_constraint()
    new_dist = constraint_to_distribution(constraint, "test")

    assert isinstance(new_dist, FloatDistribution)
    assert new_dist.name == "test"
    assert new_dist.low == -1
    assert new_dist.high == 5
