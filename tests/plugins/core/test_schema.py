# third party
import pandas as pd
import pydantic
import pytest

# synthcity absolute
from synthcity.plugins.core.schema import Schema


def test_schema_fail() -> None:
    with pytest.raises(pydantic.error_wrappers.ValidationError):
        Schema("sdfsfs")


def test_schema_ok() -> None:
    data = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
    schema = Schema(data)

    assert schema.get("a").name == "a"


def test_schema_as_constraint() -> None:
    data = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
    schema = Schema(data)

    cons = schema.as_constraint()

    assert len(cons) == 3
    for rule in cons:
        assert rule[1] == "in"


def test_schema_inclusion() -> None:
    data = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
    other = pd.DataFrame([[1, 2]], columns=["a", "b"])

    schema = Schema(data)
    other_schema = Schema(other)

    assert schema.includes(other_schema)
    assert not other_schema.includes(schema)

    data = pd.DataFrame([[1, 2, 3]], columns=["a", "d", "c"])
    other = pd.DataFrame([[1, 2]], columns=["a", "b"])

    schema = Schema(data)
    other_schema = Schema(other)

    assert not schema.includes(other_schema)
    assert not other_schema.includes(schema)

    data = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
    other = pd.DataFrame([[1, 211]], columns=["a", "b"])

    schema = Schema(data)
    other_schema = Schema(other)

    assert not schema.includes(other_schema)
    assert not other_schema.includes(schema)

    data = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
    other = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])

    schema = Schema(data)
    other_schema = Schema(other)

    assert schema.includes(other_schema)
    assert other_schema.includes(schema)
