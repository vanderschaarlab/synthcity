# stdlib
from abc import ABCMeta
from typing import Any, List, Optional

# third party
import pandas as pd
from pydantic import validate_arguments


class DataLoader(metaclass=ABCMeta):
    def __init__(self, **kwargs: Any) -> None:
        pass


class GenericDataLoader(DataLoader):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        data: pd.DataFrame,
        sensitive_columns: List[str] = [],
        target_column: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)


class SurvivalAnalysisDataLoader(DataLoader):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        data: pd.DataFrame,
        sensitive_columns: List[str] = [],
        target_column: Optional[str] = None,
        time_to_event_column: Optional[str] = None,
        time_horizons: Optional[List] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)


class TimeSeriesDataLoader(DataLoader):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
