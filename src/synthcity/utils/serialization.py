# stdlib
from pathlib import Path
from typing import Any, Union

# third party
import cloudpickle
import pandas as pd


def save(model: Any) -> bytes:
    return cloudpickle.dumps(model)


def load(buff: bytes) -> Any:
    return cloudpickle.loads(buff)


def save_to_file(path: Union[str, Path], model: Any) -> Any:
    with open(path, "wb") as f:
        return cloudpickle.dump(model, f)


def load_from_file(path: Union[str, Path]) -> Any:
    with open(path, "rb") as f:
        return cloudpickle.load(f)


def dataframe_hash(df: pd.DataFrame) -> str:
    return str(abs(pd.util.hash_pandas_object(df).sum()))
