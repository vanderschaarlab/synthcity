# third party
import numpy as np
import pandas as pd
import pydantic
import pytest
from sklearn.datasets import load_breast_cancer

# synthcity absolute
from synthcity.plugins.core.schema import Schema


def test_schema_fail() -> None:
    if pydantic.__version__ < "2":
        with pytest.raises(pydantic.error_wrappers.ValidationError):
            Schema(data="sdfsfs")
    else:
        with pytest.raises(pydantic.pydantic_core._pydantic_core.ValidationError):
            Schema(data="sdfsfs")


def test_schema_ok() -> None:
    data = pd.DataFrame(
        {
            "a": pd.Series(["a", "b", "c"], dtype="object"),
            "b": pd.Series([True, False, True], dtype="boolean"),
            "c": pd.Series([1, 2, 3], dtype="int8"),
            "d": pd.Series([4.0, 5.0, 6.0], dtype="float32"),
            "e": pd.Series([7, 8, 9], dtype="uint8"),
            "f": pd.Series(["odd", "even", "odd"], dtype="category"),
            "g": pd.Series(
                [
                    np.datetime64("2023-01-01"),
                    np.datetime64("2023-01-02"),
                    np.datetime64("2023-01-03"),
                ],
                dtype="datetime64[ns]",
            ),
        }
    )
    schema = Schema(data=data)

    assert schema.get("a").name == "a"

    assert not hasattr(schema, "data")


def test_schema_as_constraint() -> None:
    data = pd.DataFrame(
        {
            "a": pd.Series(["a", "b", "c"], dtype="object"),
            "b": pd.Series([True, False, True], dtype="boolean"),
            "c": pd.Series([1, 2, 3], dtype="int8"),
            "d": pd.Series([4.0, 5.0, 6.0], dtype="float32"),
            "e": pd.Series([7, 8, 9], dtype="uint8"),
            "f": pd.Series(["odd", "even", "odd"], dtype="category"),
            "g": pd.Series(
                [
                    np.datetime64("2023-01-01"),
                    np.datetime64("2023-01-02"),
                    np.datetime64("2023-01-03"),
                ],
                dtype="datetime64[ns]",
            ),
        }
    )
    schema = Schema(data=data)

    cons = schema.as_constraints()

    assert len(cons) == 7
    for rule in cons:
        assert rule[1] == "in"


def test_schema_from_constraint() -> None:
    data = load_breast_cancer(as_frame=True)["data"]
    schema = Schema(data=data)
    cons = schema.as_constraints()

    reloaded = Schema.from_constraints(cons)

    assert schema.domain == reloaded.domain


@pytest.mark.parametrize("sampling_strategy", ["marginal", "uniform"])
def test_schema_sample(sampling_strategy: str) -> None:
    data = load_breast_cancer(as_frame=True)["data"]
    schema = Schema(
        data=data,
        sampling_strategy=sampling_strategy,
    )

    assert schema.sample(10).shape == (10, data.shape[1])
    assert schema.sampling_strategy == sampling_strategy


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


def test_schema_dtype() -> None:
    data = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["a", "b", "c"])
    schema = Schema(data=data)

    for feature in schema:
        assert schema[feature].dtype() == "float"

    data_int = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
    schema_int = Schema(data=data_int)
    for feature in schema_int:
        assert schema_int[feature].dtype() == "int"

    data2 = schema_int.adapt_dtypes(data)
    schema2 = Schema(data=data2)

    for feature in schema2:
        assert schema2[feature].dtype() == "int"
