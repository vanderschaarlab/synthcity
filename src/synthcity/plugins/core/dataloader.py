# stdlib
import random
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# synthcity absolute
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.models.data_encoder import DatetimeEncoder
from synthcity.utils.compression import compress_dataset, decompress_dataset
from synthcity.utils.serialization import dataframe_hash


class DataLoader(metaclass=ABCMeta):
    """
    .. inheritance-diagram:: synthcity.plugins.core.dataloader.DataLoader
        :parts: 1

    Base class for all data loaders.

    Each derived class must implement the following methods:
        unpack() - a method that unpacks the columns and returns features and labels (X, y).
        decorate() - a method that creates a new instance of DataLoader by decorating the input data with the same DataLoader properties (e.g. sensitive features, target column, etc.)
        dataframe() - a method that returns the pandas dataframe that contains all features and samples
        numpy() - a method that returns the numpy array that contains all features and samples
        info() - a method that returns a dictionary of DataLoader information
        __len__() - a method that returns the number of samples in the DataLoader
        satisfies() - a method that tests if the current DataLoader satisfies the constraint provided
        match() - a method that returns a new DataLoader where the provided constraints are met
        from_info() - a static method that creates a DataLoader from the data and the information dictionary
        sample() - returns a new DataLoader that contains a random subset of N samples
        drop() - returns a new DataLoader with a list of columns dropped
        __getitem__() - getting features by names
        __setitem__() - setting features by names
        train() - returns a DataLoader containing the training set
        test() - returns a DataLoader containing the testing set
        fillna() - returns a DataLoader with NaN filled by the provided number(s)


    If any method implementation is missing, the class constructor will fail.

    Constructor Args:
        data_type: str
            The type of DataLoader, currently supports "generic", "time_series" and "survival".
        data: Any
            The object that contains the data
        static_features: List[str]
            List of feature names that are static features (as opposed to temporal features).
        temporal_features:
            List of feature names that are temporal features, i.e. observed over time.
        sensitive_features: List[str]
            Name of sensitive features.
        important_features: List[str]
            Default: None. Only relevant for SurvivalGAN method.
        outcome_features:
            The feature name that provides labels for downstream tasks.
    """

    def __init__(
        self,
        data_type: str,
        data: Any,
        static_features: List[str],
        temporal_features: List[str] = [],
        sensitive_features: List[str] = [],
        important_features: List[str] = [],
        outcome_features: List[str] = [],
        train_size: float = 0.8,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        self.static_features = static_features
        self.temporal_features = temporal_features
        self.sensitive_features = sensitive_features
        self.important_features = important_features
        self.outcome_features = outcome_features
        self.random_state = random_state

        self.data = data
        self.data_type = data_type
        self.train_size = train_size

    def raw(self) -> Any:
        return self.data

    @abstractmethod
    def unpack(self, as_numpy: bool = False, pad: bool = False) -> Any:
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
    def from_info(data: pd.DataFrame, info: dict) -> "DataLoader":
        ...

    @abstractmethod
    def sample(self, count: int, random_state: int = 0) -> "DataLoader":
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

    def hash(self) -> str:
        return dataframe_hash(self.dataframe())

    def __repr__(self, *args: Any, **kwargs: Any) -> str:
        return self.dataframe().__repr__(*args, **kwargs)

    def _repr_html_(self, *args: Any, **kwargs: Any) -> Any:
        return self.dataframe()._repr_html_(*args, **kwargs)

    @abstractmethod
    def fillna(self, value: Any) -> "DataLoader":
        ...

    @abstractmethod
    def compression_protected_features(self) -> list:
        ...

    def domain(self) -> Optional[str]:
        return None

    def compress(
        self,
    ) -> Tuple["DataLoader", Dict]:
        to_compress = self.data.copy().drop(
            columns=self.compression_protected_features()
        )
        compressed, context = compress_dataset(to_compress)
        for protected_col in self.compression_protected_features():
            compressed[protected_col] = self.data[protected_col]

        return self.decorate(compressed), context

    def decompress(self, context: Dict) -> "DataLoader":
        decompressed = decompress_dataset(self.data, context)

        return self.decorate(decompressed)

    def encode(
        self,
        encoders: Optional[Dict[str, Any]] = None,
    ) -> Tuple["DataLoader", Dict]:
        encoded = self.dataframe().copy()
        if encoders is not None:
            for col in encoders:
                if col not in encoded.columns:
                    continue
                encoded[col] = encoders[col].transform(encoded[col])
        else:
            encoders = {}

            for col in encoded.columns:
                if (
                    encoded[col].infer_objects().dtype.kind == "i"
                    and encoded[col].min() == 0
                    and encoded[col].max() == len(encoded[col].unique()) - 1
                ):
                    continue

                if (
                    encoded[col].infer_objects().dtype.kind in ["O", "b"]
                    or len(encoded[col].unique()) < 15
                ):
                    encoder = LabelEncoder().fit(encoded[col])
                    encoded[col] = encoder.transform(encoded[col])
                    encoders[col] = encoder
                elif encoded[col].infer_objects().dtype.kind in ["M"]:
                    encoder = DatetimeEncoder().fit(encoded[col])
                    encoded[col] = encoder.transform(encoded[col]).values
                    encoders[col] = encoder

        return self.from_info(encoded, self.info()), encoders

    def decode(
        self,
        encoders: Dict[str, Any],
    ) -> "DataLoader":
        decoded = self.dataframe().copy()

        for col in encoders:
            if isinstance(encoders[col], LabelEncoder):
                decoded[col] = decoded[col].astype(int)
            else:
                decoded[col] = decoded[col].astype(float)

            decoded[col] = encoders[col].inverse_transform(decoded[col])

        return self.from_info(decoded, self.info())


class GenericDataLoader(DataLoader):
    """
    .. inheritance-diagram:: synthcity.plugins.core.dataloader.GenericDataLoader
        :parts: 1

    Data loader for generic tabular data.

    Constructor Args:
        data: Union[pd.DataFrame, list, np.ndarray]
            The dataset. Either a Pandas DataFrame or a Numpy Array.
        sensitive_features: List[str]
            Name of sensitive features.
        important_features: List[str]
            Default: None. Only relevant for SurvivalGAN method.
        target_column: Optional[str]
            The feature name that provides labels for downstream tasks.
        domain_column: Optional[str]
            Optional domain label, used for domain adaptation algorithms.
        random_state: int
            Defaults to zero.

    Example:
        >>> from sklearn.datasets import load_diabetes
        >>> from synthcity.plugins.core.dataloader import GenericDataLoader
        >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
        >>> X["target"] = y
        >>> loader = GenericDataLoader(X, target_column="target", sensitive_columns=["sex"],)
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        data: Union[pd.DataFrame, list, np.ndarray],
        sensitive_features: List[str] = [],
        important_features: List[str] = [],
        target_column: Optional[str] = None,
        domain_column: Optional[str] = None,
        random_state: int = 0,
        train_size: float = 0.8,
        **kwargs: Any,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        data.columns = data.columns.astype(str)
        if target_column is not None:
            self.target_column = target_column
        elif len(data.columns) > 0:
            self.target_column = data.columns[-1]
        else:
            self.target_column = "---"

        self.domain_column = domain_column

        super().__init__(
            data_type="generic",
            data=data,
            static_features=list(data.columns),
            sensitive_features=sensitive_features,
            important_features=important_features,
            outcome_features=[self.target_column],
            random_state=random_state,
            train_size=train_size,
            **kwargs,
        )

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def domain(self) -> Optional[str]:
        return self.domain_column

    @property
    def columns(self) -> list:
        return list(self.data.columns)

    def compression_protected_features(self) -> list:
        out = [self.target_column]
        domain = self.domain()

        if domain is not None:
            out.append(domain)

        return out

    def unpack(self, as_numpy: bool = False, pad: bool = False) -> Any:
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        if as_numpy:
            return np.asarray(X), np.asarray(y)
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
            "important_features": self.important_features,
            "outcome_features": self.outcome_features,
            "target_column": self.target_column,
            "domain_column": self.domain_column,
            "train_size": self.train_size,
        }

    def __len__(self) -> int:
        return len(self.data)

    def decorate(self, data: Any) -> "DataLoader":
        return GenericDataLoader(
            data,
            sensitive_features=self.sensitive_features,
            important_features=self.important_features,
            target_column=self.target_column,
            random_state=self.random_state,
            train_size=self.train_size,
            domain_column=self.domain_column,
        )

    def satisfies(self, constraints: Constraints) -> bool:
        return constraints.is_valid(self.data)

    def match(self, constraints: Constraints) -> "DataLoader":
        return self.decorate(constraints.match(self.data))

    def sample(self, count: int, random_state: int = 0) -> "DataLoader":
        return self.decorate(self.data.sample(count, random_state=random_state))

    def drop(self, columns: list = []) -> "DataLoader":
        return self.decorate(self.data.drop(columns=columns))

    @staticmethod
    def from_info(data: pd.DataFrame, info: dict) -> "GenericDataLoader":
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Invalid data type {type(data)}")

        return GenericDataLoader(
            data,
            sensitive_features=info["sensitive_features"],
            important_features=info["important_features"],
            target_column=info["target_column"],
            domain_column=info["domain_column"],
            train_size=info["train_size"],
        )

    def __getitem__(self, feature: Union[str, list]) -> Any:
        return self.data[feature]

    def __setitem__(self, feature: str, val: Any) -> None:
        self.data[feature] = val

    def _train_test_split(self) -> Tuple:
        stratify = None
        if self.target_column in self.data:
            target = self.data[self.target_column]
            if target.value_counts().min() > 1:
                stratify = target

        return train_test_split(
            self.data,
            train_size=self.train_size,
            random_state=self.random_state,
            stratify=stratify,
        )

    def train(self) -> "DataLoader":
        train_data, _ = self._train_test_split()
        return self.decorate(train_data.reset_index(drop=True))

    def test(self) -> "DataLoader":
        _, test_data = self._train_test_split()
        return self.decorate(test_data.reset_index(drop=True))

    def fillna(self, value: Any) -> "DataLoader":
        self.data = self.data.fillna(value)
        return self


class SurvivalAnalysisDataLoader(DataLoader):
    """
    .. inheritance-diagram:: synthcity.plugins.core.dataloader.SurvivalAnalysisDataLoader
        :parts: 1

    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        data: pd.DataFrame,
        time_to_event_column: str,
        target_column: str,
        time_horizons: list = [],
        sensitive_features: List[str] = [],
        important_features: List[str] = [],
        random_state: int = 0,
        train_size: float = 0.8,
        **kwargs: Any,
    ) -> None:
        if target_column not in data.columns:
            raise ValueError(f"Event column {target_column} not found in the dataframe")

        if time_to_event_column not in data.columns:
            raise ValueError(
                f"Time to event column {time_to_event_column} not found in the dataframe"
            )

        T = data[time_to_event_column]
        data = data[T > 0]

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
            important_features=important_features,
            outcome_features=[self.target_column],
            random_state=random_state,
            train_size=train_size,
            **kwargs,
        )

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def columns(self) -> list:
        return list(self.data.columns)

    def compression_protected_features(self) -> list:
        out = [self.target_column, self.time_to_event_column]
        domain = self.domain()

        if domain is not None:
            out.append(domain)

        return out

    def unpack(self, as_numpy: bool = False, pad: bool = False) -> Any:
        X = self.data.drop(columns=[self.target_column, self.time_to_event_column])
        T = self.data[self.time_to_event_column]
        E = self.data[self.target_column]

        if as_numpy:
            return np.asarray(X), np.asarray(T), np.asarray(E)

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
            "important_features": self.important_features,
            "outcome_features": self.outcome_features,
            "target_column": self.target_column,
            "time_to_event_column": self.time_to_event_column,
            "time_horizons": self.time_horizons,
            "train_size": self.train_size,
        }

    def __len__(self) -> int:
        return len(self.data)

    def decorate(self, data: Any) -> "DataLoader":
        return SurvivalAnalysisDataLoader(
            data,
            sensitive_features=self.sensitive_features,
            important_features=self.important_features,
            target_column=self.target_column,
            time_to_event_column=self.time_to_event_column,
            time_horizons=self.time_horizons,
            random_state=self.random_state,
            train_size=self.train_size,
        )

    def satisfies(self, constraints: Constraints) -> bool:
        return constraints.is_valid(self.data)

    def match(self, constraints: Constraints) -> "DataLoader":
        return self.decorate(
            constraints.match(self.data),
        )

    def sample(self, count: int, random_state: int = 0) -> "DataLoader":
        return self.decorate(
            self.data.sample(count, random_state=random_state),
        )

    def drop(self, columns: list = []) -> "DataLoader":
        return self.decorate(
            self.data.drop(columns=columns),
        )

    @staticmethod
    def from_info(data: pd.DataFrame, info: dict) -> "DataLoader":
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Invalid data type {type(data)}")

        return SurvivalAnalysisDataLoader(
            data,
            target_column=info["target_column"],
            time_to_event_column=info["time_to_event_column"],
            sensitive_features=info["sensitive_features"],
            important_features=info["important_features"],
            time_horizons=info["time_horizons"],
        )

    def __getitem__(self, feature: Union[str, list]) -> Any:
        return self.data[feature]

    def __setitem__(self, feature: str, val: Any) -> None:
        self.data[feature] = val

    def train(self) -> "DataLoader":
        stratify = self.data[self.target_column]
        train_data, _ = train_test_split(
            self.data, train_size=self.train_size, random_state=0, stratify=stratify
        )
        return self.decorate(
            train_data.reset_index(drop=True),
        )

    def test(self) -> "DataLoader":
        stratify = self.data[self.target_column]
        _, test_data = train_test_split(
            self.data,
            train_size=self.train_size,
            random_state=0,
            stratify=stratify,
        )
        return self.decorate(
            test_data.reset_index(drop=True),
        )

    def fillna(self, value: Any) -> "DataLoader":
        self.data = self.data.fillna(value)
        return self


class TimeSeriesDataLoader(DataLoader):
    """
    .. inheritance-diagram:: synthcity.plugins.core.dataloader.TimeSeriesDataLoader
        :parts: 1
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        temporal_data: List[pd.DataFrame],
        observation_times: List,
        outcome: Optional[pd.DataFrame] = None,
        static_data: Optional[pd.DataFrame] = None,
        sensitive_features: List[str] = [],
        important_features: List[str] = [],
        random_state: int = 0,
        train_size: float = 0.8,
        seq_offset: int = 0,
        **kwargs: Any,
    ) -> None:
        static_features = []
        self.outcome_features = []

        if len(temporal_data) == 0:
            raise ValueError("Empty temporal data")

        temporal_features = TimeSeriesDataLoader.unique_temporal_features(temporal_data)

        max_window_len = max([len(t) for t in temporal_data])
        if static_data is not None:
            if len(static_data) != len(temporal_data):
                raise ValueError("Static and temporal data mismatch")
            static_features = list(static_data.columns)
        else:
            static_data = pd.DataFrame(np.zeros((len(temporal_data), 0)))

        if outcome is not None:
            if len(outcome) != len(temporal_data):
                raise ValueError("Temporal and outcome data mismatch")
            self.outcome_features = list(outcome.columns)
        else:
            outcome = pd.DataFrame(np.zeros((len(temporal_data), 0)))

        self.window_len = max_window_len
        self.fill = np.nan
        self.seq_offset = seq_offset

        (
            static_data,
            temporal_data,
            observation_times,
            outcome,
            seq_df,
            seq_info,
        ) = TimeSeriesDataLoader.pack_raw_data(
            static_data,
            temporal_data,
            observation_times,
            outcome,
            fill=self.fill,
            seq_offset=seq_offset,
        )
        self.seq_info = seq_info

        super().__init__(
            data={
                "static_data": static_data,
                "temporal_data": temporal_data,
                "observation_times": observation_times,
                "outcome": outcome,
                "seq_data": seq_df,
            },
            data_type="time_series",
            static_features=static_features,
            temporal_features=temporal_features,
            outcome_features=self.outcome_features,
            sensitive_features=sensitive_features,
            important_features=important_features,
            random_state=random_state,
            train_size=train_size,
            **kwargs,
        )

    @property
    def shape(self) -> tuple:
        return self.data["seq_data"].shape

    @property
    def columns(self) -> list:
        return self.data["seq_data"].columns

    def compression_protected_features(self) -> list:
        return self.outcome_features

    @property
    def raw_columns(self) -> list:
        return self.static_features + self.temporal_features + self.outcome_features

    def dataframe(self) -> pd.DataFrame:
        return self.data["seq_data"].copy()

    def numpy(self) -> np.ndarray:
        return self.dataframe().values

    def info(self) -> dict:
        generic_info = {
            "data_type": self.data_type,
            "len": len(self),
            "static_features": self.static_features,
            "temporal_features": self.temporal_features,
            "outcome_features": self.outcome_features,
            "outcome_len": len(self.data["outcome"].values.reshape(-1))
            / len(self.data["outcome"]),
            "window_len": self.window_len,
            "sensitive_features": self.sensitive_features,
            "important_features": self.important_features,
            "random_state": self.random_state,
            "train_size": self.train_size,
            "fill": self.fill,
        }

        for key in self.seq_info:
            generic_info[key] = self.seq_info[key]

        return generic_info

    def __len__(self) -> int:
        return len(self.data["seq_data"])

    def decorate(self, data: Any) -> "DataLoader":
        static_data, temporal_data, observation_times, outcome = data

        return TimeSeriesDataLoader(
            temporal_data,
            observation_times=observation_times,
            static_data=static_data,
            outcome=outcome,
            sensitive_features=self.sensitive_features,
            important_features=self.important_features,
            random_state=self.random_state,
            train_size=self.train_size,
            seq_offset=self.seq_offset,
        )

    def unpack_and_decorate(self, data: pd.DataFrame) -> "DataLoader":
        unpacked_data = TimeSeriesDataLoader.unpack_raw_data(
            data,
            self.info(),
        )

        return self.decorate(unpacked_data)

    def satisfies(self, constraints: Constraints) -> bool:
        seq_df = self.dataframe()

        return constraints.is_valid(seq_df)

    def match(self, constraints: Constraints) -> "DataLoader":
        return self.unpack_and_decorate(
            constraints.match(self.dataframe()),
        )

    def drop(self, columns: list = []) -> "DataLoader":
        new_data = self.data["seq_data"].drop(columns=columns)
        return self.unpack_and_decorate(new_data)

    @staticmethod
    def from_info(data: pd.DataFrame, info: dict) -> "DataLoader":
        (
            static_data,
            temporal_data,
            observation_times,
            outcome,
        ) = TimeSeriesDataLoader.unpack_raw_data(
            data,
            info,
        )
        return TimeSeriesDataLoader(
            temporal_data,
            observation_times=observation_times,
            static_data=static_data,
            outcome=outcome,
            sensitive_features=info["sensitive_features"],
            important_features=info["important_features"],
            fill=info["fill"],
            seq_offset=info["seq_offset"],
        )

    def unpack(self, as_numpy: bool = False, pad: bool = False) -> Any:
        if pad:
            (
                static_data,
                temporal_data,
                observation_times,
                outcome,
            ) = TimeSeriesDataLoader.pad_and_mask(
                self.data["static_data"],
                self.data["temporal_data"],
                self.data["observation_times"],
                self.data["outcome"],
            )
        else:
            static_data, temporal_data, observation_times, outcome = (
                self.data["static_data"],
                self.data["temporal_data"],
                self.data["observation_times"],
                self.data["outcome"],
            )
        if as_numpy:
            return (
                np.asarray(static_data),
                np.asarray(temporal_data),
                np.asarray(observation_times),
                np.asarray(outcome),
            )
        return (
            static_data,
            temporal_data,
            observation_times,
            outcome,
        )

    def __getitem__(self, feature: Union[str, list]) -> Any:
        return self.data["seq_data"][feature]

    def __setitem__(self, feature: str, val: Any) -> None:
        self.data["seq_data"][feature] = val

    def ids(self) -> list:
        id_col = self.seq_info["seq_id_feature"]
        ids = self.data["seq_data"][id_col]

        return list(ids.unique())

    def filter_ids(self, ids_list: list) -> pd.DataFrame:
        seq_data = self.data["seq_data"]
        id_col = self.info()["seq_id_feature"]

        return seq_data[seq_data[id_col].isin(ids_list)]

    def train(self) -> "DataLoader":
        # TODO: stratify
        ids = self.ids()
        train_ids, _ = train_test_split(
            ids,
            train_size=self.train_size,
            random_state=self.random_state,
        )
        return self.unpack_and_decorate(self.filter_ids(train_ids))

    def test(self) -> "DataLoader":
        # TODO: stratify
        ids = self.ids()
        _, test_ids = train_test_split(
            ids,
            train_size=self.train_size,
            random_state=self.random_state,
        )
        return self.unpack_and_decorate(self.filter_ids(test_ids))

    def sample(self, count: int, random_state: int = 0) -> "DataLoader":
        ids = self.ids()
        count = min(count, len(ids))
        sampled_ids = random.sample(ids, count)

        return self.unpack_and_decorate(self.filter_ids(sampled_ids))

    def fillna(self, value: Any) -> "DataLoader":
        for key in ["static_data", "outcome", "seq_data"]:
            if self.data[key] is not None:
                self.data[key] = self.data[key].fillna(value)

        for idx, item in enumerate(self.data["temporal_data"]):
            self.data["temporal_data"][idx] = self.data["temporal_data"][idx].fillna(
                value
            )

        return self

    @staticmethod
    def unique_temporal_features(temporal_data: List[pd.DataFrame]) -> List:
        temporal_features = []
        for item in temporal_data:
            temporal_features.extend(item.columns)
        return sorted(np.unique(temporal_features).tolist())

    # Padding helpers
    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def pad_raw_features(
        static_data: Optional[pd.DataFrame],
        temporal_data: List[pd.DataFrame],
        observation_times: List,
        outcome: Optional[pd.DataFrame],
    ) -> Any:
        fill = np.nan

        temporal_features = TimeSeriesDataLoader.unique_temporal_features(temporal_data)

        for idx, item in enumerate(temporal_data):
            # handling missing features
            for col in temporal_features:
                if col not in item.columns:
                    item[col] = fill
            item = item[temporal_features]

            if list(item.columns) != list(temporal_features):
                raise RuntimeError("Invalid features for packing")

            temporal_data[idx] = item.fillna(fill)

        return static_data, temporal_data, observation_times, outcome

    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def pad_raw_data(
        static_data: Optional[pd.DataFrame],
        temporal_data: List[pd.DataFrame],
        observation_times: List,
        outcome: Optional[pd.DataFrame],
    ) -> Any:
        fill = np.nan

        (
            static_data,
            temporal_data,
            observation_times,
            outcome,
        ) = TimeSeriesDataLoader.pad_raw_features(
            static_data, temporal_data, observation_times, outcome
        )
        max_window_len = max([len(t) for t in temporal_data])
        temporal_features = TimeSeriesDataLoader.unique_temporal_features(temporal_data)

        for idx, item in enumerate(temporal_data):
            if len(item) != max_window_len:
                pads = fill * np.ones(
                    (max_window_len - len(item), len(temporal_features))
                )
                start = 0
                if len(item.index) > 0:
                    start = max(item.index) + 1
                pads_df = pd.DataFrame(
                    pads,
                    index=[start + i for i in range(len(pads))],
                    columns=item.columns,
                )
                item = pd.concat([item, pads_df])

            # handle missing time points
            if list(item.columns) != list(temporal_features):
                raise RuntimeError(
                    f"Invalid features {item.columns}. Expected {temporal_features}"
                )
            if len(item) != max_window_len:
                raise RuntimeError("Invalid window len")

            temporal_data[idx] = item

        observation_times_padded = []
        for idx, item in enumerate(observation_times):
            item = list(item)
            if len(item) != max_window_len:
                pads = fill * np.ones(max_window_len - len(item))
                item.extend(pads.tolist())
            observation_times_padded.append(item)

        return static_data, temporal_data, observation_times_padded, outcome

    # Masking helpers
    @staticmethod
    def extract_masked_features(full_temporal_features: list) -> tuple:
        temporal_features = []
        mask_features = []
        mask_prefix = "masked_"
        for feat in full_temporal_features:
            feat = str(feat)
            if not feat.startswith(mask_prefix):
                temporal_features.append(feat)
                continue

            other_feat = feat[len(mask_prefix) :]
            if other_feat in full_temporal_features:
                mask_features.append(feat)
            else:
                temporal_features.append(feat)

        return temporal_features, mask_features

    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def mask_temporal_data(
        temporal_data: List[pd.DataFrame],
        observation_times: List,
        fill: Any = 0,
    ) -> Any:
        nan_cnt = 0
        for item in temporal_data:
            nan_cnt += np.asarray(np.isnan(item)).sum()

        if nan_cnt == 0:
            return temporal_data, observation_times

        temporal_features = TimeSeriesDataLoader.unique_temporal_features(temporal_data)
        masked_features = [f"masked_{feat}" for feat in temporal_features]

        for idx, item in enumerate(temporal_data):
            item[masked_features] = (~np.isnan(item)).astype(int)
            item = item.fillna(fill)
            temporal_data[idx] = item

        for idx, item in enumerate(observation_times):
            item = np.nan_to_num(item, nan=fill).tolist()

            observation_times[idx] = item

        return temporal_data, observation_times

    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def unmask_temporal_data(
        temporal_data: List[pd.DataFrame],
        observation_times: List,
        fill: Any = np.nan,
    ) -> Any:
        temporal_features = TimeSeriesDataLoader.unique_temporal_features(temporal_data)
        temporal_features, mask_features = TimeSeriesDataLoader.extract_masked_features(
            temporal_features
        )

        missing_horizons = []
        for idx, item in enumerate(temporal_data):
            # handle existing mask
            if len(mask_features) > 0:
                mask = temporal_data[idx][mask_features].astype(bool)
                item[~mask] = np.nan

            item_missing_rows = item.isna().sum(axis=1).values
            missing_horizons.append(item_missing_rows == len(temporal_features))

            # TODO: review impact on horizons
            temporal_data[idx] = item.dropna()

        observation_times_unmasked = []
        for idx, item in enumerate(observation_times):
            item = list(item)

            for midx, mval in enumerate(missing_horizons[idx]):
                if mval:
                    item[midx] = np.nan

            local_horizons = list(filter(lambda v: v == v, item))
            observation_times_unmasked.append(local_horizons)

        return temporal_data, observation_times_unmasked

    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def pad_and_mask(
        static_data: Optional[pd.DataFrame],
        temporal_data: List[pd.DataFrame],
        observation_times: List,
        outcome: Optional[pd.DataFrame],
        only_features: Any = False,
        fill: Any = 0,
    ) -> Any:
        if only_features:
            (
                static_data,
                temporal_data,
                observation_times,
                outcome,
            ) = TimeSeriesDataLoader.pad_raw_features(
                static_data,
                temporal_data,
                observation_times,
                outcome,
            )
        else:
            (
                static_data,
                temporal_data,
                observation_times,
                outcome,
            ) = TimeSeriesDataLoader.pad_raw_data(
                static_data,
                temporal_data,
                observation_times,
                outcome,
            )

        temporal_data, observation_times = TimeSeriesDataLoader.mask_temporal_data(
            temporal_data, observation_times, fill=fill
        )

        return static_data, temporal_data, observation_times, outcome

    @staticmethod
    def sequential_view(
        static_data: Optional[pd.DataFrame],
        temporal_data: List[pd.DataFrame],
        observation_times: List,
        outcome: Optional[pd.DataFrame],
        id_col: str = "seq_id",
        time_id_col: str = "seq_time_id",
        seq_offset: int = 0,
    ) -> Tuple[pd.DataFrame, dict]:  # sequential dataframe, loader info
        (
            static_data,
            temporal_data,
            observation_times,
            outcome,
        ) = TimeSeriesDataLoader.pad_and_mask(
            static_data, temporal_data, observation_times, outcome, only_features=True
        )
        raw_static_features = list(static_data.columns)
        static_features = [f"seq_static_{col}" for col in raw_static_features]

        raw_outcome_features = list(outcome.columns)
        outcome_features = [f"seq_out_{col}" for col in raw_outcome_features]

        raw_temporal_features = TimeSeriesDataLoader.unique_temporal_features(
            temporal_data
        )
        temporal_features = [f"seq_temporal_{col}" for col in raw_temporal_features]
        cols = (
            [id_col, time_id_col]
            + static_features
            + temporal_features
            + outcome_features
        )

        seq = []
        for sidx, static_item in static_data.iterrows():
            real_tidx = 0
            for tidx, temporal_item in temporal_data[sidx].iterrows():
                local_seq_data = (
                    [
                        sidx + seq_offset,
                        observation_times[sidx][real_tidx],
                    ]
                    + static_item[raw_static_features].values.tolist()
                    + temporal_item[raw_temporal_features].values.tolist()
                    + outcome.loc[sidx, raw_outcome_features].values.tolist()
                )
                seq.append(local_seq_data)
                real_tidx += 1

        seq_df = pd.DataFrame(seq, columns=cols)
        info = {
            "seq_static_features": static_features,
            "seq_temporal_features": temporal_features,
            "seq_outcome_features": outcome_features,
            "seq_offset": seq_offset,
            "seq_id_feature": id_col,
            "seq_time_id_feature": time_id_col,
            "seq_features": list(seq_df.columns),
        }
        return seq_df, info

    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def pack_raw_data(
        static_data: Optional[pd.DataFrame],
        temporal_data: List[pd.DataFrame],
        observation_times: List,
        outcome: Optional[pd.DataFrame],
        fill: Any = np.nan,
        seq_offset: int = 0,
    ) -> pd.DataFrame:

        # Temporal data: (subjects, temporal_sequence, temporal_feature)
        temporal_features = TimeSeriesDataLoader.unique_temporal_features(temporal_data)
        temporal_features, mask_features = TimeSeriesDataLoader.extract_masked_features(
            temporal_features
        )
        temporal_data, observation_times = TimeSeriesDataLoader.unmask_temporal_data(
            temporal_data, observation_times
        )
        seq_df, info = TimeSeriesDataLoader.sequential_view(
            static_data=static_data,
            temporal_data=temporal_data,
            observation_times=observation_times,
            outcome=outcome,
            seq_offset=seq_offset,
        )

        return static_data, temporal_data, observation_times, outcome, seq_df, info

    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def unpack_raw_data(
        data: pd.DataFrame,
        info: dict,
    ) -> Tuple[
        Optional[pd.DataFrame], List[pd.DataFrame], List, Optional[pd.DataFrame]
    ]:
        id_col = info["seq_id_feature"]
        time_col = info["seq_time_id_feature"]

        static_cols = info["seq_static_features"]
        new_static_cols = [feat.split("seq_static_")[1] for feat in static_cols]

        temporal_cols = info["seq_temporal_features"]
        new_temporal_cols = [feat.split("seq_temporal_")[1] for feat in temporal_cols]

        outcome_cols = info["seq_outcome_features"]
        new_outcome_cols = [feat.split("seq_out_")[1] for feat in outcome_cols]

        ids = sorted(list(set(data[id_col])))

        static_data = []
        temporal_data = []
        observation_times = []
        outcome_data = []

        for item_id in ids:
            item_data = data[data[id_col] == item_id]

            static_data.append(item_data[static_cols].head(1).values.squeeze().tolist())
            outcome_data.append(
                item_data[outcome_cols].head(1).values.squeeze().tolist()
            )
            local_temporal_data = item_data[temporal_cols].copy()
            local_observation_times = item_data[time_col].values.tolist()
            local_temporal_data.columns = new_temporal_cols
            # TODO: review impact on horizons
            local_temporal_data = local_temporal_data.dropna()

            temporal_data.append(local_temporal_data)
            observation_times.append(local_observation_times)

        static_df = pd.DataFrame(static_data, columns=new_static_cols)
        outcome_df = pd.DataFrame(outcome_data, columns=new_outcome_cols)

        return static_df, temporal_data, observation_times, outcome_df


class TimeSeriesSurvivalDataLoader(TimeSeriesDataLoader):
    """
    .. inheritance-diagram:: synthcity.plugins.core.dataloader.TimeSeriesSurvivalDataLoader
        :parts: 1
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        temporal_data: List[pd.DataFrame],
        observation_times: Union[List, np.ndarray, pd.Series],
        T: Union[pd.Series, np.ndarray, pd.Series],
        E: Union[pd.Series, np.ndarray, pd.Series],
        static_data: Optional[pd.DataFrame] = None,
        sensitive_features: List[str] = [],
        important_features: List[str] = [],
        time_horizons: list = [],
        random_state: int = 0,
        train_size: float = 0.8,
        seq_offset: int = 0,
        **kwargs: Any,
    ) -> None:
        self.time_to_event_col = "time_to_event"
        self.event_col = "event"

        if len(time_horizons) == 0:
            time_horizons = np.linspace(T.min(), T.max(), num=5)[1:-1].tolist()
        self.time_horizons = time_horizons
        outcome = pd.concat([pd.Series(T), pd.Series(E)], axis=1)
        outcome.columns = [self.time_to_event_col, self.event_col]

        self.fill = np.nan

        super().__init__(
            temporal_data=temporal_data,
            observation_times=observation_times,
            outcome=outcome,
            static_data=static_data,
            sensitive_features=sensitive_features,
            important_features=important_features,
            random_state=random_state,
            train_size=train_size,
            seq_offset=seq_offset,
            **kwargs,
        )
        self.data_type = "time_series_survival"

    def info(self) -> dict:
        parent_info = super().info()
        parent_info["time_to_event_column"] = self.time_to_event_col
        parent_info["event_column"] = self.event_col
        parent_info["time_horizons"] = self.time_horizons
        parent_info["fill"] = self.fill

        return parent_info

    def decorate(self, data: Any) -> "DataLoader":
        static_data, temporal_data, observation_times, outcome = data
        if self.time_to_event_col not in outcome:
            raise ValueError(
                f"Survival outcome is missing tte column {self.time_to_event_col}"
            )
        if self.event_col not in outcome:
            raise ValueError(
                f"Survival outcome is missing event column {self.event_col}"
            )

        return TimeSeriesSurvivalDataLoader(
            temporal_data,
            observation_times=observation_times,
            static_data=static_data,
            T=outcome[self.time_to_event_col],
            E=outcome[self.event_col],
            sensitive_features=self.sensitive_features,
            important_features=self.important_features,
            random_state=self.random_state,
            time_horizons=self.time_horizons,
            train_size=self.train_size,
            seq_offset=self.seq_offset,
        )

    @staticmethod
    def from_info(data: pd.DataFrame, info: dict) -> "DataLoader":
        (
            static_data,
            temporal_data,
            observation_times,
            outcome,
        ) = TimeSeriesSurvivalDataLoader.unpack_raw_data(
            data,
            info,
        )
        return TimeSeriesSurvivalDataLoader(
            temporal_data,
            observation_times=observation_times,
            static_data=static_data,
            T=outcome[info["time_to_event_column"]],
            E=outcome[info["event_column"]],
            sensitive_features=info["sensitive_features"],
            important_features=info["important_features"],
            time_horizons=info["time_horizons"],
            seq_offset=info["seq_offset"],
        )

    def unpack(self, as_numpy: bool = False, pad: bool = False) -> Any:
        if pad:
            (
                static_data,
                temporal_data,
                observation_times,
                outcome,
            ) = TimeSeriesSurvivalDataLoader.pad_and_mask(
                self.data["static_data"],
                self.data["temporal_data"],
                self.data["observation_times"],
                self.data["outcome"],
            )
        else:
            static_data, temporal_data, observation_times, outcome = (
                self.data["static_data"],
                self.data["temporal_data"],
                self.data["observation_times"],
                self.data["outcome"],
            )

        if as_numpy:
            return (
                np.asarray(static_data),
                np.asarray(temporal_data, dtype=object),
                np.asarray(observation_times, dtype=object),
                np.asarray(outcome[self.time_to_event_col]),
                np.asarray(outcome[self.event_col]),
            )
        return (
            static_data,
            temporal_data,
            observation_times,
            outcome[self.time_to_event_col],
            outcome[self.event_col],
        )

    def match(self, constraints: Constraints) -> "DataLoader":
        return self.unpack_and_decorate(
            constraints.match(self.dataframe()),
        )

    def train(self) -> "DataLoader":
        stratify = self.data["outcome"][self.event_col]

        ids = self.ids()
        train_ids, _ = train_test_split(
            ids,
            train_size=self.train_size,
            random_state=self.random_state,
            stratify=stratify,
        )
        return self.unpack_and_decorate(self.filter_ids(train_ids))

    def test(self) -> "DataLoader":
        stratify = self.data["outcome"][self.event_col]
        ids = self.ids()
        _, test_ids = train_test_split(
            ids,
            train_size=self.train_size,
            random_state=self.random_state,
            stratify=stratify,
        )
        return self.unpack_and_decorate(self.filter_ids(test_ids))


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def create_from_info(data: pd.DataFrame, info: dict) -> "DataLoader":
    if info["data_type"] == "generic":
        return GenericDataLoader.from_info(data, info)
    elif info["data_type"] == "survival_analysis":
        return SurvivalAnalysisDataLoader.from_info(data, info)
    elif info["data_type"] == "time_series":
        return TimeSeriesDataLoader.from_info(data, info)
    elif info["data_type"] == "time_series_survival":
        return TimeSeriesSurvivalDataLoader.from_info(data, info)
    else:
        raise RuntimeError(f"invalid datatype {info}")
