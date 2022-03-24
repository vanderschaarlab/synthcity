# third party
import pandas as pd
import pydantic
import pytest
from sklearn.datasets import load_breast_cancer

# synthcity absolute
from synthcity.plugins.core.schema import Schema


def test_schema_fail() -> None:
    with pytest.raises(pydantic.error_wrappers.ValidationError):
        Schema(data="sdfsfs")


def test_schema_ok() -> None:
    data = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
    schema = Schema(data=data)

    assert schema.get("a").name == "a"

    assert not hasattr(schema, "data")


def test_schema_as_constraint() -> None:
    data = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
    schema = Schema(data=data)

    cons = schema.as_constraints()

    assert len(cons) == 3
    for rule in cons:
        assert rule[1] == "in"


def test_schema_from_constraint() -> None:
    data = load_breast_cancer(as_frame=True)["data"]
    schema = Schema(data=data)
    cons = schema.as_constraints()

    reloaded = Schema.from_constraints(cons)

    assert schema.domain == reloaded.domain


@pytest.mark.parametrize("dp_enabled", [False, True])
@pytest.mark.parametrize("dp_epsilon", [1, 10])
@pytest.mark.parametrize("sampling_strategy", ["marginal", "uniform"])
def test_schema_sample(
    dp_enabled: float, dp_epsilon: float, sampling_strategy: str
) -> None:
    data = load_breast_cancer(as_frame=True)["data"]
    schema = Schema(
        data=data,
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon,
        sampling_strategy=sampling_strategy,
    )

    assert schema.sample(10).shape == (10, data.shape[1])
    assert schema.sampling_strategy == sampling_strategy
    assert schema.dp_epsilon == dp_epsilon
    assert schema.dp_enabled == dp_enabled


def test_schema_inclusion() -> None:
    data = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
    other = pd.DataFrame([[1, 2]], columns=["a", "b"])

    schema = Schema(data=data)
    other_schema = Schema(data=other)

    assert schema.includes(other_schema)
    assert not other_schema.includes(schema)

    data = pd.DataFrame([[1, 2, 3]], columns=["a", "d", "c"])
    other = pd.DataFrame([[1, 2]], columns=["a", "b"])

    schema = Schema(data=data)
    other_schema = Schema(data=other)

    assert not schema.includes(other_schema)
    assert not other_schema.includes(schema)

    data = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
    other = pd.DataFrame([[1, 211]], columns=["a", "b"])

    schema = Schema(data=data)
    other_schema = Schema(data=other)

    assert not schema.includes(other_schema)
    assert not other_schema.includes(schema)

    data = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
    other = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])

    schema = Schema(data=data)
    other_schema = Schema(data=other)

    assert schema.includes(other_schema)
    assert other_schema.includes(schema)
