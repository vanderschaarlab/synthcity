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

    @abstractmethod
    def preprocessed(self) -> Any:
        ...

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
    def sample(self, count: int) -> "DataLoader":
        ...

    @abstractmethod
    def __getitem__(self, feature: str) -> Any:
        ...

    @abstractmethod
    def __setitem__(self, feature: str, val: Any) -> None:
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

    def preprocessed(self) -> Any:
        return self.data

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

    def sample(self, count: int) -> "DataLoader":
        return GenericDataLoader(
            self.data.sample(count), sensitive_features=self.sensitive_features
        )

    @staticmethod
    def from_info(data: Any, info: dict) -> "GenericDataLoader":
        assert isinstance(data, pd.DataFrame)

        return GenericDataLoader(data, sensitive_features=info["sensitive_features"])

    def __getitem__(self, feature: str) -> Any:
        return self.data[feature]

    def __setitem__(self, feature: str, val: Any) -> None:
        self.data[feature] = val


class SurvivalAnalysisDataLoader(DataLoader):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        data: pd.DataFrame,
        time_to_event_column: str,
        target_column: str,
        sensitive_features: List[str] = [],
        **kwargs: Any,
    ) -> None:
        if target_column not in data.columns:
            raise ValueError(f"Event column {target_column} not found in the dataframe")

        if time_to_event_column not in data.columns:
            raise ValueError(
                f"Time to event column {time_to_event_column} not found in the dataframe"
            )

        data.columns = data.columns.astype(str)

        self.target_column = target_column
        self.time_to_event_column = time_to_event_column

        super().__init__(
            data_type="survival_analysis",
            data=data,
            static_features=list(data.columns.astype(str)),
            sensitive_features=sensitive_features,
            **kwargs,
        )

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def preprocessed(self) -> Any:
        X = self.data.drop(columns=[self.target_column, self.time_to_event_column])
        T = self.data[self.time_to_event_column]
        E = self.data[self.target_column]

        return X, T, E

    def dataframe(self) -> pd.DataFrame:
        return self.data

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
        return len(self.data)

    def satisfies(self, constraints: Constraints) -> bool:
        return constraints.is_valid(self.data)

    def match(self, constraints: Constraints) -> "DataLoader":
        return SurvivalAnalysisDataLoader(
            constraints.match(self.data),
            sensitive_features=self.sensitive_features,
            target_column=self.target_column,
            time_to_event_column=self.time_to_event_column,
        )

    def sample(self, count: int) -> "DataLoader":
        return SurvivalAnalysisDataLoader(
            self.data.sample(count),
            sensitive_features=self.sensitive_features,
            target_column=self.target_column,
            time_to_event_column=self.time_to_event_column,
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

    def __setitem__(self, feature: str, val: Any) -> None:
        self.data[feature] = val


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

    def sample(self, count: int) -> "DataLoader":
        return TimeSeriesDataLoader(
            self.data.sample(count), sensitive_features=self.sensitive_features
        )

    @staticmethod
    def from_info(data: Any, info: dict) -> "DataLoader":
        pass

    def preprocessed(self) -> Any:
        return self.data

    def __getitem__(self, feature: str) -> Any:
        return self.data[feature]

    def __setitem__(self, feature: str, val: Any) -> None:
        self.data[feature] = val


def create_from_info(data: Any, info: dict) -> "DataLoader":
    if info["data_type"] == "generic":
        return GenericDataLoader.from_info(data, info)
    elif info["data_type"] == "survival_analysis":
        return SurvivalAnalysisDataLoader.from_info(data, info)
    elif info["data_type"] == "time_series":
        return TimeSeriesDataLoader.from_info(data, info)
    else:
        raise RuntimeError(f"invalid datatype {info}")
