# stdlib
from typing import Any

# third party
import pandas as pd
from pydantic import validate_arguments


class DatetimeEncoder:
    def __init__(self) -> None:
        pass

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: pd.Series) -> Any:
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def transform(self, X: pd.Series) -> pd.Series:
        out = pd.to_numeric(X).astype(float)
        return out

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def inverse_transform(self, X: pd.Series) -> pd.Series:
        out = pd.to_datetime(X)
        return out

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit_transform(self, X: pd.Series) -> pd.Series:
        return self.fit(X).transform(X)
