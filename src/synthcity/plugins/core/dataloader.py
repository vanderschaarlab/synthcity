# stdlib
from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Tuple, Union

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.model_selection import train_test_split

# synthcity absolute
from synthcity.plugins.core.constraints import Constraints
from synthcity.utils.serialization import dataframe_hash


class DataLoader(metaclass=ABCMeta):
    def __init__(
        self,
        data_type: str,
        data: Any,
        static_features: List[str],
        temporal_features: List[str] = [],
        sensitive_features: List[str] = [],
        outcome_features: List[str] = [],
        train_size: float = 0.8,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        self.static_features = static_features
        self.temporal_features = temporal_features
        self.sensitive_features = sensitive_features
        self.outcome_features = outcome_features
        self.random_state = random_state

        self.data = data
        self.data_type = data_type
        self.train_size = train_size

    def raw(self) -> Any:
        return self.data

    @abstractmethod
    def unpack(self, as_numpy: bool = False) -> Any:
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

    def hash(self) -> str:
        return dataframe_hash(self.dataframe())

    def __repr__(self, *args: Any, **kwargs: Any) -> str:
        return self.dataframe().__repr__(*args, **kwargs)

    def _repr_html_(self, *args: Any, **kwargs: Any) -> Any:
        return self.dataframe()._repr_html_(*args, **kwargs)

    @abstractmethod
    def fillna(self, value: Any) -> "DataLoader":
        ...

    def sequential_view(
        self,
    ) -> Tuple[pd.DataFrame, dict]:  # sequential dataframe, info
        raise NotImplementedError()

    @staticmethod
    def from_sequential_view(seq_df: pd.DataFrame, info: dict) -> "DataLoader":
        raise NotImplementedError()


class GenericDataLoader(DataLoader):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        data: Union[pd.DataFrame, list, np.ndarray],
        sensitive_features: List[str] = [],
        target_column: Optional[str] = None,
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

        super().__init__(
            data_type="generic",
            data=data,
            static_features=list(data.columns),
            sensitive_features=sensitive_features,
            outcome_features=[self.target_column],
            random_state=random_state,
            **kwargs,
        )

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def columns(self) -> list:
        return list(self.data.columns)

    def unpack(self, as_numpy: bool = False) -> Any:
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
            "outcome_features": self.outcome_features,
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
            random_state=self.random_state,
            train_size=self.train_size,
        )

    def match(self, constraints: Constraints) -> "DataLoader":
        return self.decorate(constraints.match(self.data))

    def sample(self, count: int) -> "DataLoader":
        return self.decorate(self.data.sample(count))

    def drop(self, columns: list = []) -> "DataLoader":
        return self.decorate(self.data.drop(columns=columns))

    @staticmethod
    def from_info(data: pd.DataFrame, info: dict) -> "GenericDataLoader":
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

    def fillna(self, value: Any) -> "DataLoader":
        self.data = self.data.fillna(value)
        return self


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
            outcome_features=[self.target_column],
            random_state=random_state,
            **kwargs,
        )

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def columns(self) -> list:
        return list(self.data.columns)

    def unpack(self, as_numpy: bool = False) -> Any:
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
            "outcome_features": self.outcome_features,
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
            random_state=self.random_state,
            train_size=self.train_size,
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
    def from_info(data: pd.DataFrame, info: dict) -> "DataLoader":
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

    def fillna(self, value: Any) -> "DataLoader":
        self.data = self.data.fillna(value)
        return self


class TimeSeriesDataLoader(DataLoader):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        temporal_data: List[pd.DataFrame],
        outcome: Optional[pd.DataFrame] = None,
        static_data: Optional[pd.DataFrame] = None,
        sensitive_features: List[str] = [],
        random_state: int = 0,
        train_size: float = 0.8,
        fill: Any = np.nan,
        **kwargs: Any,
    ) -> None:
        static_features = []
        self.outcome_features = []

        if len(temporal_data) == 0:
            raise ValueError("Empty temporal data")

        temporal_features = TimeSeriesDataLoader.unique_temporal_features(temporal_data)

        max_seq_len = max([len(t) for t in temporal_data])
        if static_data is not None:
            if len(static_data) != len(temporal_data):
                raise ValueError("Static and temporal data mismatch")
            static_features = list(static_data.columns)

        if outcome is not None:
            if len(outcome) != len(temporal_data):
                raise ValueError("Temporal and outcome data mismatch")
            self.outcome_features = list(outcome.columns)

        self.seq_len = max_seq_len
        self.fill = fill

        grouped = TimeSeriesDataLoader.pack_raw_data(
            static_data, temporal_data, outcome, fill=fill
        )

        super().__init__(
            data={
                "static_data": static_data,
                "temporal_data": temporal_data,
                "outcome": outcome,
                "grouped_data": grouped,
            },
            data_type="time_series",
            static_features=static_features,
            temporal_features=temporal_features,
            outcome_features=self.outcome_features,
            sensitive_features=sensitive_features,
            random_state=random_state,
            **kwargs,
        )

    @staticmethod
    def unique_temporal_features(temporal_data: List[pd.DataFrame]) -> List:
        temporal_features = []
        for item in temporal_data:
            temporal_features.extend(item.columns)
        return sorted(np.unique(temporal_features).tolist())

    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def pack_raw_data(
        static_data: Optional[pd.DataFrame],
        temporal_data: List[pd.DataFrame],
        outcome: Optional[pd.DataFrame],
        fill: Any = np.nan,
    ) -> pd.DataFrame:
        # Temporal data: (subjects, temporal_sequence, temporal_feature)
        ext_temporal_features = []
        max_seq_len = max([len(t) for t in temporal_data])
        temporal_features = TimeSeriesDataLoader.unique_temporal_features(temporal_data)

        for feat in temporal_features:
            ext_temporal_features.extend(
                [f"temporal_{feat}_t{idx}" for idx in range(max_seq_len)]
            )

        for idx, item in enumerate(temporal_data):
            # handling missing features
            for col in temporal_features:
                if col not in item.columns:
                    item[col] = fill
            item = item[temporal_features]
            if len(item) != max_seq_len:
                pads = fill * np.ones((max_seq_len - len(item), len(temporal_features)))
                start = max(item.index) + 1
                pads_df = pd.DataFrame(
                    pads,
                    index=[start + i for i in range(len(pads))],
                    columns=item.columns,
                )
                item = pd.concat([item, pads_df])

            # handle missing time points
            assert list(item.columns) == list(temporal_features)
            assert len(item) == max_seq_len

            temporal_data[idx] = item

        temporal_arr = np.asarray(temporal_data)
        temporal_arr = np.swapaxes(temporal_arr, 1, 2)

        out_df = pd.DataFrame(
            temporal_arr.reshape(len(temporal_arr), -1), columns=ext_temporal_features
        )
        if static_data is not None:
            out_df = pd.concat(
                [
                    pd.DataFrame(
                        static_data.values,
                        columns=[f"static_{feat}" for feat in static_data.columns],
                    ),
                    out_df,
                ],
                axis=1,
            )
        if outcome is not None:
            out_df = pd.concat(
                [
                    out_df,
                    pd.DataFrame(
                        outcome.values,
                        columns=[f"out_{feat}" for feat in outcome.columns],
                    ),
                ],
                axis=1,
            )
        for col in out_df.columns:
            if "int" in str(out_df[col].dtype):
                out_df[col] = out_df[col].astype(int)
            if "float" in str(out_df[col].dtype):
                out_df[col] = out_df[col].astype(float)

        return out_df

    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def unpack_raw_data(
        data: pd.DataFrame,
        static_features: List[str],
        temporal_features: List[str],
        outcome_features: List[str],
        seq_len: int,
    ) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, Optional[pd.DataFrame]]:
        static_data: Optional[pd.DataFrame] = pd.DataFrame(np.zeros((len(data), 0)))
        if len(static_features) > 0:
            static_data = pd.DataFrame([], columns=static_features, index=data.index)
            for feat in static_features:
                static_data[feat] = data[f"static_{feat}"]

        outcome: Optional[pd.DataFrame] = pd.DataFrame(np.zeros((len(data), 0)))
        if len(outcome_features) > 0:
            outcome = pd.DataFrame([], columns=outcome_features, index=data.index)
            for feat in outcome_features:
                outcome[feat] = data[f"out_{feat}"]

        temporal_data = []
        for idx, row in data.iterrows():
            local_df = pd.DataFrame(
                [], columns=temporal_features, index=list(range(seq_len))
            )
            for feat in temporal_features:
                for seq in range(seq_len):
                    local_df.loc[seq, feat] = row[f"temporal_{feat}_t{seq}"]
                local_df[feat] = local_df[feat].astype(
                    row[f"temporal_{feat}_t{seq}"].dtype
                )
            temporal_data.append(local_df[temporal_features].dropna())
        return static_data, temporal_data, outcome

    @property
    def shape(self) -> tuple:
        return self.data["grouped_data"].shape

    @property
    def columns(self) -> list:
        return self.data["grouped_data"].columns

    @property
    def raw_columns(self) -> list:
        return self.static_features + self.temporal_features + self.outcome_features

    def dataframe(self) -> pd.DataFrame:
        return self.data["grouped_data"]

    def numpy(self) -> np.ndarray:
        return self.dataframe().values

    def info(self) -> dict:
        return {
            "data_type": self.data_type,
            "len": len(self),
            "static_features": self.static_features,
            "temporal_features": self.temporal_features,
            "group_features": list(self.data["grouped_data"].columns),
            "outcome_features": self.outcome_features,
            "outcome_len": len(self.data["outcome"].values.reshape(-1))
            / len(self.data["outcome"]),
            "seq_len": self.seq_len,
            "sensitive_features": self.sensitive_features,
            "random_state": self.random_state,
            "train_size": self.train_size,
            "fill": self.fill,
        }

    def __len__(self) -> int:
        return len(self.data["grouped_data"])

    def decorate(self, data: Any) -> "DataLoader":
        static_data, temporal_data, outcome = data

        return TimeSeriesDataLoader(
            temporal_data,
            static_data=static_data,
            outcome=outcome,
            sensitive_features=self.sensitive_features,
            random_state=self.random_state,
            train_size=self.train_size,
        )

    def unpack_and_decorate(self, data: pd.DataFrame) -> "DataLoader":
        unpacked_data = TimeSeriesDataLoader.unpack_raw_data(
            data,
            self.static_features,
            self.temporal_features,
            self.outcome_features,
            self.seq_len,
        )

        return self.decorate(unpacked_data)

    def satisfies(self, constraints: Constraints) -> bool:
        return constraints.is_valid(self.data["grouped_data"])

    def match(self, constraints: Constraints) -> "DataLoader":
        new_data = constraints.match(self.data["grouped_data"])

        return self.unpack_and_decorate(new_data)

    def sample(self, count: int) -> "DataLoader":
        new_data = self.data["grouped_data"].sample(count)
        return self.unpack_and_decorate(new_data)

    def drop(self, columns: list = []) -> "DataLoader":
        new_data = self.data["grouped_data"].drop(columns=columns)
        return self.unpack_and_decorate(new_data)

    @staticmethod
    def from_info(data: pd.DataFrame, info: dict) -> "DataLoader":
        static_data, temporal_data, outcome = TimeSeriesDataLoader.unpack_raw_data(
            data,
            static_features=info["static_features"],
            temporal_features=info["temporal_features"],
            outcome_features=info["outcome_features"],
            seq_len=info["seq_len"],
        )
        return TimeSeriesDataLoader(
            temporal_data,
            static_data=static_data,
            outcome=outcome,
            sensitive_features=info["sensitive_features"],
        )

    def unpack(self, as_numpy: bool = False) -> Any:
        if as_numpy:
            return (
                np.asarray(self.data["static_data"]),
                np.asarray(self.data["temporal_data"]),
                np.asarray(self.data["outcome"]),
            )
        return (
            self.data["static_data"],
            self.data["temporal_data"],
            self.data["outcome"],
        )

    def __getitem__(self, feature: Union[str, list]) -> Any:
        return self.data["grouped_data"][feature]

    def __setitem__(self, feature: str, val: Any) -> None:
        self.data["grouped_data"][feature] = val

    def train(self) -> "DataLoader":
        new_data, _ = train_test_split(
            self.data["grouped_data"],
            train_size=self.train_size,
            random_state=self.random_state,
        )
        return self.unpack_and_decorate(new_data)

    def test(self) -> "DataLoader":
        _, new_data = train_test_split(
            self.data["grouped_data"],
            train_size=self.train_size,
            random_state=self.random_state,
        )
        return self.unpack_and_decorate(new_data)

    def fillna(self, value: Any) -> "DataLoader":
        for key in ["static_data", "outcome", "grouped_data"]:
            if self.data[key] is not None:
                self.data[key] = self.data[key].fillna(value)

        for idx, item in enumerate(self.data["temporal_data"]):
            self.data["temporal_data"][idx] = self.data["temporal_data"][idx].fillna(
                value
            )

        return self

    def sequential_view(
        self,
    ) -> Tuple[pd.DataFrame, dict]:  # sequential dataframe, loader info
        id_col = "id"
        time_col = "time_id"
        static_features = [f"static_{col}" for col in self.static_features]
        outcome_features = [f"out_{col}" for col in self.outcome_features]
        temporal_features = [f"temporal_{col}" for col in self.temporal_features]
        cols = (
            [id_col, time_col] + static_features + temporal_features + outcome_features
        )

        seq = []
        for sidx, static_item in self.data["static_data"].iterrows():
            for tidx, temporal_item in self.data["temporal_data"][sidx].iterrows():
                local_seq_data = (
                    [sidx, tidx]
                    + static_item[self.static_features].values.tolist()
                    + temporal_item[self.temporal_features].values.tolist()
                    + self.data["outcome"]
                    .loc[sidx, self.outcome_features]
                    .values.tolist()
                )
                seq.append(local_seq_data)

        seq_df = pd.DataFrame(seq, columns=cols)
        info = self.info()
        info["id_feature"] = id_col
        info["time_feature"] = time_col
        info["static_features"] = static_features
        info["temporal_features"] = temporal_features
        info["outcome_features"] = outcome_features

        return seq_df, info

    @staticmethod
    def from_sequential_view_prepare(
        seq_df: pd.DataFrame,
        info: dict,
    ) -> Tuple:
        id_col = info["id_feature"]
        time_col = info["time_feature"]

        static_cols = info["static_features"]
        new_static_cols = [feat.split("static_")[1] for feat in static_cols]

        temporal_cols = info["temporal_features"]
        new_temporal_cols = [feat.split("temporal_")[1] for feat in temporal_cols]

        outcome_cols = info["outcome_features"]
        new_outcome_cols = [feat.split("out_")[1] for feat in outcome_cols]

        ids = sorted(list(set(seq_df[id_col])))

        static_data = []
        temporal_data = []
        outcome_data = []

        for item_id in ids:
            item_data = seq_df[seq_df[id_col] == item_id]

            static_data.append(item_data[static_cols].head(1).values.squeeze().tolist())
            outcome_data.append(
                item_data[outcome_cols].head(1).values.squeeze().tolist()
            )
            local_temporal_data = item_data[temporal_cols].copy()
            local_temporal_data.index = item_data[time_col]
            local_temporal_data.columns = new_temporal_cols
            local_temporal_data = local_temporal_data.dropna()

            temporal_data.append(local_temporal_data)

        static_df = pd.DataFrame(static_data, columns=new_static_cols, index=ids)
        outcome_df = pd.DataFrame(outcome_data, columns=new_outcome_cols, index=ids)

        return static_df, temporal_data, outcome_df

    @staticmethod
    def from_sequential_view(
        seq_df: pd.DataFrame,
        info: dict,
    ) -> "DataLoader":
        (
            static_df,
            temporal_data,
            outcome_df,
        ) = TimeSeriesDataLoader.from_sequential_view_prepare(seq_df, info)

        return TimeSeriesDataLoader(
            temporal_data=temporal_data,
            outcome=outcome_df,
            static_data=static_df,
            sensitive_features=info["sensitive_features"],
            random_state=info["random_state"],
            train_size=info["train_size"],
            fill=info["fill"],
        )


class TimeSeriesSurvivalDataLoader(TimeSeriesDataLoader):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        temporal_data: List[pd.DataFrame],
        T: pd.Series,
        E: pd.Series,
        static_data: Optional[pd.DataFrame] = None,
        sensitive_features: List[str] = [],
        time_horizons: list = [],
        random_state: int = 0,
        train_size: float = 0.8,
        fill: Any = np.nan,
        **kwargs: Any,
    ) -> None:
        outcome = pd.concat([T, E], axis=1)
        self.time_to_event_col = "time_to_event"
        self.event_col = "event"

        if len(time_horizons) == 0:
            time_horizons = np.linspace(T.min(), T.max(), num=5)[1:-1].tolist()
        self.time_horizons = time_horizons

        super().__init__(
            temporal_data=temporal_data,
            outcome=outcome,
            static_data=static_data,
            sensitive_features=sensitive_features,
            random_state=random_state,
            train_size=train_size,
            fill=fill,
            **kwargs,
        )
        self.data_type = "time_series_survival"
        self.fill = fill

    def info(self) -> dict:
        parent_info = super().info()
        parent_info["time_to_event_col"] = self.time_to_event_col
        parent_info["event_col"] = self.event_col
        parent_info["time_horizons"] = self.time_horizons
        parent_info["fill"] = self.fill

        return parent_info

    def decorate(self, data: Any) -> "DataLoader":
        static_data, temporal_data, outcome = data
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
            static_data=static_data,
            T=outcome[self.time_to_event_col],
            E=outcome[self.event_col],
            sensitive_features=self.sensitive_features,
            random_state=self.random_state,
            time_horizons=self.time_horizons,
            train_size=self.train_size,
        )

    @staticmethod
    def from_info(data: pd.DataFrame, info: dict) -> "DataLoader":
        (
            static_data,
            temporal_data,
            outcome,
        ) = TimeSeriesSurvivalDataLoader.unpack_raw_data(
            data,
            static_features=info["static_features"],
            temporal_features=info["temporal_features"],
            outcome_features=info["outcome_features"],
            seq_len=info["seq_len"],
        )
        return TimeSeriesSurvivalDataLoader(
            temporal_data,
            static_data=static_data,
            T=outcome[info["time_to_event_col"]],
            E=outcome[info["event_col"]],
            sensitive_features=info["sensitive_features"],
            time_horizons=info["time_horizons"],
        )

    def unpack(self, as_numpy: bool = False) -> Any:
        if as_numpy:
            return (
                np.asarray(self.data["static_data"]),
                np.asarray(self.data["temporal_data"], dtype=object),
                np.asarray(self.data["outcome"][self.time_to_event_col]),
                np.asarray(self.data["outcome"][self.event_col]),
            )
        return (
            self.data["static_data"],
            self.data["temporal_data"],
            self.data["outcome"][self.time_to_event_col],
            self.data["outcome"][self.event_col],
        )

    @staticmethod
    def from_sequential_view(
        seq_df: pd.DataFrame,
        info: dict,
    ) -> "DataLoader":
        (
            static_df,
            temporal_data,
            outcome_df,
        ) = TimeSeriesSurvivalDataLoader.from_sequential_view_prepare(seq_df, info)

        return TimeSeriesSurvivalDataLoader(
            temporal_data=temporal_data,
            T=outcome_df[info["time_to_event_col"]],
            E=outcome_df[info["event_col"]],
            static_data=static_df,
            sensitive_features=info["sensitive_features"],
            time_horizons=info["time_horizons"],
            random_state=info["random_state"],
            train_size=info["train_size"],
            fill=info["fill"],
        )


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
