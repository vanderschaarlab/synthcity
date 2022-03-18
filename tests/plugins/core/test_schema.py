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
