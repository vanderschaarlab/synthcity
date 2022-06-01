# stdlib
from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Union

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
        train_size: float = 0.8,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        self.static_features = static_features
        self.dynamic_features = dynamic_features
        self.sensitive_features = sensitive_features
        self.random_state = random_state

        self.data = data
        self.data_type = data_type
        self.train_size = train_size

    def raw(self) -> Any:
        return self.data

    @abstractmethod
    def unpack(self) -> Any:
        ...

    @abstractmethod
    def decorate(self, data: Any) -> "DataLoader":
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
    def drop(self, columns: list = []) -> "DataLoader":
        ...

    @abstractmethod
    def __getitem__(self, feature: Union[str, list]) -> Any:
        ...

    @abstractmethod
    def __setitem__(self, feature: str, val: Any) -> None:
        ...

    @abstractmethod
    def train(self) -> "DataLoader":
        ...

    @abstractmethod
    def test(self) -> "DataLoader":
        ...

    @abstractmethod
    def normalize(self) -> "DataLoader":
        ...


class GenericDataLoader(DataLoader):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        data: Union[pd.DataFrame, list, np.ndarray],
        sensitive_features: List[str] = [],
        target_column: Optional[str] = None,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        data.columns = data.columns.astype(str)
        if target_column is not None:
            self.target_column = target_column
        else:
            self.target_column = data.columns[-1]

        super().__init__(
            data_type="generic",
            data=data,
            static_features=list(data.columns),
            sensitive_features=sensitive_features,
            random_state=random_state,
            **kwargs,
        )

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def columns(self) -> list:
        return list(self.data.columns)

    def unpack(self) -> Any:
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        return X, y

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
            "target_column": self.target_column,
        }

    def __len__(self) -> int:
        return len(self.data)

    def satisfies(self, constraints: Constraints) -> bool:
        return constraints.is_valid(self.data)

    def decorate(self, data: Any) -> "DataLoader":
        return GenericDataLoader(
            data,
            sensitive_features=self.sensitive_features,
            target_column=self.target_column,
        )

    def match(self, constraints: Constraints) -> "DataLoader":
        return self.decorate(constraints.match(self.data))

    def sample(self, count: int) -> "DataLoader":
        return self.decorate(self.data.sample(count))

    def drop(self, columns: list = []) -> "DataLoader":
        return self.decorate(self.data.drop(columns=columns))

    @staticmethod
    def from_info(data: Any, info: dict) -> "GenericDataLoader":
        assert isinstance(data, pd.DataFrame)

        return GenericDataLoader(
            data,
            sensitive_features=info["sensitive_features"],
            target_column=info["target_column"],
        )

    def __getitem__(self, feature: Union[str, list]) -> Any:
        return self.data[feature]

    def __setitem__(self, feature: str, val: Any) -> None:
        self.data[feature] = val

    def train(self) -> "DataLoader":
        train_data, _ = train_test_split(
            self.data, train_size=self.train_size, random_state=self.random_state
        )
        return self.decorate(train_data.reset_index(drop=True))

    def test(self) -> "DataLoader":
        _, test_data = train_test_split(
            self.data, train_size=self.train_size, random_state=self.random_state
        )
        return self.decorate(test_data.reset_index(drop=True))

    def normalize(self) -> "DataLoader":
        norm = MinMaxScaler().fit_transform(self.data)

        return self.decorate(norm)


class SurvivalAnalysisDataLoader(DataLoader):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        data: pd.DataFrame,
        time_to_event_column: str,
        target_column: str,
        time_horizons: list = [],
        sensitive_features: List[str] = [],
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        if target_column not in data.columns:
            raise ValueError(f"Event column {target_column} not found in the dataframe")

        if time_to_event_column not in data.columns:
            raise ValueError(
                f"Time to event column {time_to_event_column} not found in the dataframe"
            )

        T = data[time_to_event_column]
        if len(time_horizons) == 0:
            time_horizons = np.linspace(T.min(), T.max(), num=5)[1:-1].tolist()

        data.columns = data.columns.astype(str)

        self.target_column = target_column
        self.time_to_event_column = time_to_event_column
        self.time_horizons = time_horizons

        super().__init__(
            data_type="survival_analysis",
            data=data,
            static_features=list(data.columns.astype(str)),
            sensitive_features=sensitive_features,
            random_state=random_state,
            **kwargs,
        )

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def columns(self) -> list:
        return list(self.data.columns)

    def unpack(self) -> Any:
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
            "time_horizons": self.time_horizons,
        }

    def __len__(self) -> int:
        return len(self.data)

    def satisfies(self, constraints: Constraints) -> bool:
        return constraints.is_valid(self.data)

    def decorate(self, data: Any) -> "DataLoader":
        return SurvivalAnalysisDataLoader(
            data,
            sensitive_features=self.sensitive_features,
            target_column=self.target_column,
            time_to_event_column=self.time_to_event_column,
            time_horizons=self.time_horizons,
        )

    def match(self, constraints: Constraints) -> "DataLoader":
        return self.decorate(
            constraints.match(self.data),
        )

    def sample(self, count: int) -> "DataLoader":
        return self.decorate(
            self.data.sample(count),
        )

    def drop(self, columns: list = []) -> "DataLoader":
        return self.decorate(
            self.data.drop(columns=columns),
        )

    @staticmethod
    def from_info(data: Any, info: dict) -> "DataLoader":
        assert isinstance(data, pd.DataFrame)

        return SurvivalAnalysisDataLoader(
            data,
            target_column=info["target_column"],
            time_to_event_column=info["time_to_event_column"],
            sensitive_features=info["sensitive_features"],
            time_horizons=info["time_horizons"],
        )

    def __getitem__(self, feature: Union[str, list]) -> Any:
        return self.data[feature]

    def __setitem__(self, feature: str, val: Any) -> None:
        self.data[feature] = val

    def train(self) -> "DataLoader":
        train_data, _ = train_test_split(
            self.data, train_size=self.train_size, random_state=0
        )
        return self.decorate(
            train_data.reset_index(drop=True),
        )

    def test(self) -> "DataLoader":
        _, test_data = train_test_split(
            self.data, train_size=self.train_size, random_state=0
        )
        return self.decorate(
            test_data.reset_index(drop=True),
        )

    def normalize(self) -> "DataLoader":
        X, T, E = self.unpack()
        norm = MinMaxScaler().fit_transform(X)

        norm[self.target_column] = E
        norm[self.time_to_event_column] = T

        return self.decorate(norm)


class TimeSeriesDataLoader(DataLoader):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        static_data: pd.DataFrame,
        temporal_data: List[pd.DataFrame],
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_type="time_series", random_state=random_state, **kwargs)

    @property
    def shape(self) -> tuple:
        raise NotImplementedError()

    @property
    def columns(self) -> list:
        raise NotImplementedError()

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
        raise NotImplementedError()

    def decorate(self, data: Any) -> "DataLoader":
        raise NotImplementedError()

    def match(self, constraints: Constraints) -> "DataLoader":
        raise NotImplementedError()

    def sample(self, count: int) -> "DataLoader":
        raise NotImplementedError()

    def drop(self, columns: list = []) -> "DataLoader":
        raise NotImplementedError()

    @staticmethod
    def from_info(data: Any, info: dict) -> "DataLoader":
        raise NotImplementedError()

    def unpack(self) -> Any:
        raise NotImplementedError()

    def __getitem__(self, feature: Union[str, list]) -> Any:
        raise NotImplementedError()

    def __setitem__(self, feature: str, val: Any) -> None:
        raise NotImplementedError()

    def train(self) -> "DataLoader":
        raise NotImplementedError()

    def test(self) -> "DataLoader":
        raise NotImplementedError()

    def normalize(self) -> "DataLoader":
        raise NotImplementedError()


def create_from_info(data: Any, info: dict) -> "DataLoader":
    if info["data_type"] == "generic":
        return GenericDataLoader.from_info(data, info)
    elif info["data_type"] == "survival_analysis":
        return SurvivalAnalysisDataLoader.from_info(data, info)
    elif info["data_type"] == "time_series":
        return TimeSeriesDataLoader.from_info(data, info)
    else:
        raise RuntimeError(f"invalid datatype {info}")
