# stdlib
from abc import ABCMeta, abstractmethod
from typing import Any, List, Union

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.constraints import Constraints


class DataLoader(metaclass=ABCMeta):
    def __init__(
        self,
        data_type: str,
        data: Any,
        static_features: List[str],
        dynamic_features: List[str] = [],
        sensitive_features: List[str] = [],
        **kwargs: Any,
    ) -> None:
        self.static_features = static_features
        self.dynamic_features = dynamic_features
        self.sensitive_features = sensitive_features

        self.data = data
        self.data_type = data_type

    def raw(self) -> Any:
        return self.data

    def type(self) -> str:
        return self.data_type

    @property
    def shape(self) -> tuple:
        ...

    @property
    def columns(self) -> list:
        ...

    @abstractmethod
    def dataframe(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def numpy(self) -> np.ndarray:
        ...

    @property
    def values(self) -> np.ndarray:
        return self.numpy()

    @abstractmethod
    def info(self) -> dict:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def satisfies(self, constraints: Constraints) -> bool:
        ...

    @abstractmethod
    def match(self, constraints: Constraints) -> "DataLoader":
        ...

    @staticmethod
    @abstractmethod
    def from_info(data: Any, info: dict) -> "DataLoader":
        ...

    @abstractmethod
    def __getitem__(self, feature: str) -> Any:
        ...


class GenericDataLoader(DataLoader):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        data: Union[pd.DataFrame, list, np.ndarray],
        sensitive_features: List[str] = [],
        **kwargs: Any,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        data.columns = data.columns.astype(str)
        super().__init__(
            data_type="generic",
            data=data,
            static_features=list(data.columns),
            sensitive_features=sensitive_features,
            **kwargs,
        )

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def columns(self) -> list:
        return list(self.data.columns)

    def dataframe(self) -> pd.DataFrame:
        return self.data

    def numpy(self) -> np.ndarray:
        return self.dataframe().values

    def info(self) -> dict:
        return {
            "data_type": self.data_type,
            "len": len(self),
            "static_features": self.static_features,
            "sensitive_features": self.sensitive_features,
        }

    def __len__(self) -> int:
        return len(self.data)

    def satisfies(self, constraints: Constraints) -> bool:
        return constraints.is_valid(self.data)

    def match(self, constraints: Constraints) -> "DataLoader":
        return GenericDataLoader(
            constraints.match(self.data), sensitive_features=self.sensitive_features
        )

    @staticmethod
    def from_info(data: Any, info: dict) -> "GenericDataLoader":
        assert isinstance(data, pd.DataFrame)

        return GenericDataLoader(data, sensitive_features=info["sensitive_features"])

    def __getitem__(self, feature: str) -> Any:
        return self.data[feature]


class SurvivalAnalysisDataLoader(DataLoader):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        data: pd.DataFrame,
        time_to_event_column: str,
        target_column: str,
        time_horizons: List,
        sensitive_features: List[str] = [],
        **kwargs: Any,
    ) -> None:
        data.columns = data.columns.astype(str)

        X = data.drop(columns=[target_column, time_to_event_column])
        T = data[time_to_event_column]
        E = data[target_column]

        self.target_column = target_column
        self.time_to_event_column = time_to_event_column

        super().__init__(
            data_type="survival_analysis",
            data=(X, T, E),
            static_features=list(X.columns.astype(str)),
            sensitive_features=sensitive_features,
            **kwargs,
        )

    @property
    def shape(self) -> tuple:
        return self.data[0].shape

    def dataframe(self) -> pd.DataFrame:
        X, T, E = self.data

        df = X.copy()
        df[self.time_to_event_column] = T
        df[self.target_column] = E

        return df

    def numpy(self) -> np.ndarray:
        return self.dataframe().values

    def info(self) -> dict:
        return {
            "data_type": self.data_type,
            "len": len(self),
            "static_features": list(self.static_features),
            "sensitive_features": self.sensitive_features,
            "target_column": self.target_column,
            "time_to_event_column": self.time_to_event_column,
        }

    def __len__(self) -> int:
        return len(self.data[0])

    def satisfies(self, constraints: Constraints) -> bool:
        return constraints.is_valid(self.data)

    def match(self, constraints: Constraints) -> "DataLoader":
        return SurvivalAnalysisDataLoader(
            constraints.match(self.data), sensitive_features=self.sensitive_features
        )

    @staticmethod
    def from_info(data: Any, info: dict) -> "DataLoader":
        assert isinstance(data, pd.DataFrame)

        return SurvivalAnalysisDataLoader(
            data,
            target_column=info["target_column"],
            time_to_event_column=info["time_to_event_column"],
            sensitive_features=info["sensitive_features"],
        )

    def __getitem__(self, feature: str) -> Any:
        return self.data[feature]


class TimeSeriesDataLoader(DataLoader):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        static_data: pd.DataFrame,
        temporal_data: List[pd.DataFrame],
        **kwargs: Any,
    ) -> None:
        super().__init__(data_type="time_series", **kwargs)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def dataframe(self) -> pd.DataFrame:
        raise NotImplementedError("TimeSeries type not supported for dataframes")

    def numpy(self) -> np.ndarray:
        return self.dataframe().values

    def info(self) -> dict:
        return {
            "data_type": self.data_type,
            "len": len(self),
            "static_features": self.static_features,
            "sensitive_features": self.sensitive_features,
        }

    def __len__(self) -> int:
        return len(self.data)

    def satisfies(self, constraints: Constraints) -> bool:
        return constraints.is_valid(self.data)

    def match(self, constraints: Constraints) -> "DataLoader":
        return constraints.match(self.data)

    @staticmethod
    def from_info(data: Any, info: dict) -> "DataLoader":
        pass


def create_from_info(data: Any, info: dict) -> "DataLoader":
    if info["data_type"] == "generic":
        return GenericDataLoader.from_info(data, info)
    elif info["data_type"] == "survival_analysis":
        return SurvivalAnalysisDataLoader.from_info(data, info)
    elif info["data_type"] == "time_series":
        return TimeSeriesDataLoader.from_info(data, info)
    else:
        raise RuntimeError(f"invalid datatype {info}")
