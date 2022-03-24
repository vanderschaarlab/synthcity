# third party
import pandas as pd
import pytest

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

    assert param.marginal_distribution is None


@pytest.mark.parametrize("use_dp", [True, False])
def test_categorical_from_data(use_dp: bool) -> None:
    param = CategoricalDistribution(
        use_dp=use_dp,
        name="test",
        data=pd.Series([1, 1, 1, 1, 2, 2, 2, 22, 3, 3, 3, 3]),
    )

    assert set(param.get()[1]) == set([1, 2, 3, 22])
    assert len(param.sample(count=5)) == 5
    for sample in param.sample(count=5):
        assert sample in [1, 2, 3, 22]

    assert param.has(1)
    assert not param.has(5)
    assert len(param.as_constraint().rules) == 1

    param_other = CategoricalDistribution(name="test", choices=[1, 2])
    assert param.includes(param_other)
    assert not param_other.includes(param)

    assert param.marginal_distribution is not None
    assert set(param.marginal_distribution.keys()) == set([1, 2, 3, 22])


def test_categorical_invalid_epsi_dp() -> None:
    with pytest.raises(ValueError):
        CategoricalDistribution(
            use_dp=True,
            epsilon=0,
            name="test",
            data=pd.Series([1, 1, 1, 1, 2, 2, 2, 22, 3, 3, 3, 3]),
        )


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

    assert param.marginal_distribution is None


@pytest.mark.parametrize("use_dp", [True, False])
def test_integer_from_data(use_dp: bool) -> None:
    param = IntegerDistribution(
        use_dp=use_dp, name="test", data=pd.Series([1, 1, 1, 12, 2, 2, 2, 4, 4, 88, 4])
    )

    assert param.get() == ["test", 1, 88, 1]
    assert len(param.sample(count=5)) == 5
    for sample in param.sample(count=5):
        assert sample in list(range(0, 101))
    assert param.has(1)
    assert not param.has(101)
    assert not param.has(-1)
    assert len(param.as_constraint().rules) == 3

    param_other = IntegerDistribution(name="test", low=1, high=77)
    assert param.includes(param_other)
    assert not param_other.includes(param)

    assert param.marginal_distribution is not None
    assert set(param.marginal_distribution.keys()) == set([1, 2, 4, 12, 88])


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

    assert param.marginal_distribution is None


@pytest.mark.parametrize("use_dp", [True, False])
def test_float_from_data(use_dp: bool) -> None:
    param = FloatDistribution(
        use_dp=use_dp,
        name="test",
        data=pd.Series([0, 1.1, 2.3, 1, 0.5, 1, 1, 1, 1, 1, 1]),
    )

    assert param.get() == ["test", 0, 2.3]
    assert len(param.sample(count=5)) == 5
    for sample in param.sample(count=5):
        assert sample <= 2.3
        assert sample >= 0
    assert len(param.as_constraint().rules) == 3

    param_other = FloatDistribution(name="test", low=0, high=0.5)
    assert param.includes(param_other)
    assert not param_other.includes(param)

    assert param.marginal_distribution is not None
    assert set(param.marginal_distribution.keys()) == set([0, 1.1, 2.3, 1.0, 0.5])


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
