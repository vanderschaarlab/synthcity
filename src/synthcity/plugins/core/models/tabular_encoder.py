"""TabularEncoder module.

Adapted from https://github.com/sdv-dev/CTGAN
"""

# stdlib
import platform
from collections import namedtuple
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from rdt.transformers import ClusterBasedNormalizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# synthcity absolute
import synthcity.logger as log
from synthcity.utils.serialization import dataframe_hash, load_from_file, save_to_file

ColumnTransformInfo = namedtuple(
    "ColumnTransformInfo",
    ["column_name", "column_type", "transform", "output_dimensions"],
)


class BinEncoder(TransformerMixin, BaseEstimator):
    """Binary encoder (for SurvivalGAN).

    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        max_clusters: int = 10,
        categorical_limit: int = 10,
    ) -> None:
        """Create a data transformer.

        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
        """
        self.max_clusters = max_clusters
        self.categorical_limit = categorical_limit

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _fit_continuous(self, data: pd.Series) -> ColumnTransformInfo:
        """Train Bayesian GMM for continuous columns.

        Args:
            data (pd.Series):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.name
        gm = ClusterBasedNormalizer(
            model_missing_values=True,
            max_clusters=min(self.max_clusters, len(data)),
            enforce_min_max_values=True,
        )
        gm.fit(data.to_frame(), [column_name])
        num_components = sum(gm.valid_component_indicator)

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="continuous",
            transform=gm,
            output_dimensions=1 + num_components,
        )

    def fit(
        self, raw_data: pd.Series, discrete_columns: Optional[List] = None
    ) -> "BinEncoder":
        """Fit the ``BinEncoder``.

        Fits a ``ClusterBasedNormalizer`` for continuous columns
        """
        if discrete_columns is None:
            discrete_columns = []

            for col in raw_data.columns:
                if len(raw_data[col].unique()) < self.categorical_limit:
                    discrete_columns.append(col)

        self.output_dimensions = 0

        self._column_transform_info = {}
        for column_name in raw_data.columns:
            if column_name not in discrete_columns:
                column_transform_info = self._fit_continuous(raw_data[column_name])
                self._column_transform_info[column_name] = column_transform_info

        return self

    def _transform_continuous(
        self, column_transform_info: ColumnTransformInfo, data: pd.Series
    ) -> pd.Series:
        column_name = data.name
        gm = column_transform_info.transform
        transformed = gm.transform(data.to_frame(), [column_name])

        return transformed[f"{column_name}.component"].to_numpy().astype(int)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def transform(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Take raw data and output a matrix data."""
        output = raw_data.copy()

        for column_name in self._column_transform_info:
            column_transform_info = self._column_transform_info[column_name]

            output[column_name] = self._transform_continuous(
                column_transform_info, raw_data[column_name]
            )

        return output

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit_transform(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(raw_data).transform(raw_data)


class TabularEncoder(TransformerMixin, BaseEstimator):
    """Tabular encoder.

    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        max_clusters: int = 10,
        categorical_limit: int = 10,
        whitelist: list = [],
        workspace: Path = Path("workspace"),
    ) -> None:
        """Create a data transformer.

        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
        """
        self.max_clusters = max_clusters
        self.categorical_limit = categorical_limit
        self.whitelist = whitelist
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _fit_continuous(self, data: pd.Series) -> ColumnTransformInfo:
        """Train Bayesian GMM for continuous columns.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.name
        gm = ClusterBasedNormalizer(
            model_missing_values=True,
            max_clusters=min(len(data), self.max_clusters),
            enforce_min_max_values=True,
        )
        gm.fit(data.to_frame(), [column_name])
        num_components = sum(gm.valid_component_indicator)

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="continuous",
            transform=gm,
            output_dimensions=1 + num_components,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _fit_discrete(self, data: pd.Series) -> ColumnTransformInfo:
        """Fit one hot encoder for discrete column.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.name
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        ohe.fit(data.values.reshape(-1, 1))
        num_categories = len(ohe.categories_[0])

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="discrete",
            transform=ohe,
            output_dimensions=num_categories,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self, raw_data: pd.DataFrame, discrete_columns: Optional[List] = None
    ) -> Any:
        """Fit the ``TabularEncoder``.

        Fits a ``ClusterBasedNormalizer`` for continuous columns and a
        ``OneHotEncoder`` for discrete columns.

        This step also counts the #columns in matrix data and span information.
        """
        if discrete_columns is None:
            discrete_columns = []

            for col in raw_data.columns:
                if len(raw_data[col].unique()) < self.categorical_limit:
                    discrete_columns.append(col)

        self.output_dimensions = 0

        self._column_raw_dtypes = raw_data.infer_objects().dtypes
        self._column_transform_info_list = []

        self.workspace.mkdir(parents=True, exist_ok=True)

        for column_name in raw_data.columns:
            if column_name in self.whitelist:
                continue
            column_hash = dataframe_hash(raw_data[[column_name]])
            bkp_file = (
                self.workspace
                / f"encoder_cache_{column_hash}_{column_name[:50]}_{self.max_clusters}_{self.categorical_limit}_{platform.python_version()}.bkp"
            )

            log.info(f"Encoding {column_name} {column_hash}")

            if bkp_file.exists():
                column_transform_info = load_from_file(bkp_file)
            else:
                if column_name in discrete_columns:
                    column_transform_info = self._fit_discrete(raw_data[column_name])
                else:
                    column_transform_info = self._fit_continuous(raw_data[column_name])
                save_to_file(bkp_file, column_transform_info)

            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

        return self

    def _transform_continuous(
        self, column_transform_info: ColumnTransformInfo, data: pd.Series
    ) -> pd.DataFrame:
        column_name = data.name
        gm = column_transform_info.transform
        transformed = gm.transform(data.to_frame(), [column_name])

        #  Converts the transformed data to the appropriate output format.
        #  The first column (ending in '.normalized') stays the same,
        #  but the lable encoded column (ending in '.component') is one hot encoded.
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed[f"{column_name}.normalized"].to_numpy()
        index = transformed[f"{column_name}.component"].to_numpy().astype(int)
        output[np.arange(index.size), index + 1] = 1

        return pd.DataFrame(
            output,
            columns=[f"{column_name}.normalized"]
            + [
                f"{column_name}.component_{i}"
                for i in range(column_transform_info.output_dimensions - 1)
            ],
        )

    def _transform_discrete(
        self, column_transform_info: ColumnTransformInfo, data: pd.Series
    ) -> pd.DataFrame:
        ohe = column_transform_info.transform
        return pd.DataFrame(
            ohe.transform(data.to_frame().values),
            columns=ohe.get_feature_names_out([data.name]),
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def transform(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Take raw data and output a matrix data."""
        if len(self._column_transform_info_list) == 0:
            return pd.DataFrame(np.zeros((len(raw_data), 0)))

        column_data_list = []
        for column_name in self.whitelist:
            if column_name not in raw_data.columns:
                continue
            data = raw_data[column_name]
            column_data_list.append(data)

        for column_transform_info in self._column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[column_name]

            if column_transform_info.column_type == "continuous":
                column_data_list.append(
                    self._transform_continuous(column_transform_info, data)
                )
            else:
                column_data_list.append(
                    self._transform_discrete(column_transform_info, data)
                )

        result = pd.concat(column_data_list, axis=1)
        result.index = raw_data.index

        return result

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _inverse_transform_continuous(
        self,
        column_transform_info: ColumnTransformInfo,
        column_data: pd.DataFrame,
    ) -> pd.DataFrame:
        gm = column_transform_info.transform
        data = pd.DataFrame(
            column_data.values[:, :2], columns=list(gm.get_output_sdtypes())
        )
        data.iloc[:, 1] = np.argmax(column_data.values[:, 1:], axis=1)
        return gm.reverse_transform(data, [column_transform_info.column_name])

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _inverse_transform_discrete(
        self, column_transform_info: ColumnTransformInfo, column_data: pd.DataFrame
    ) -> pd.DataFrame:
        ohe = column_transform_info.transform
        column = column_transform_info.column_name
        return pd.DataFrame(
            ohe.inverse_transform(column_data),
            columns=[column],
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        """
        if len(self._column_transform_info_list) == 0:
            return pd.DataFrame(np.zeros((len(data), 0)))

        st = 0
        recovered_column_data_list = []
        column_names = []
        column_types = []

        for column_name in self.whitelist:
            if column_name not in data.columns:
                continue
            local_data = data[column_name]
            column_names.append(column_name)
            column_types.append(self._column_raw_dtypes)
            recovered_column_data_list.append(local_data)

        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data.iloc[:, list(range(st, st + dim))]
            if column_transform_info.column_type == "continuous":
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data
                )
            else:
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data
                )

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = pd.DataFrame(
            recovered_data, columns=column_names, index=data.index
        ).astype(self._column_raw_dtypes.filter(column_names))
        return recovered_data

    def layout(self) -> List[Tuple]:
        """Get the layout of the encoded dataset.

        Returns a list of tuple, describing each column as:
            - continuous, and with length 1 + number of GMM clusters.
            - discrete, and with length <N>, the length of the one-hot encoding.
        """
        return self._column_transform_info_list

    def n_features(self) -> int:
        return np.sum(
            [
                column_transform_info.output_dimensions
                for column_transform_info in self._column_transform_info_list
            ]
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def activation_layout(
        self, discrete_activation: str, continuous_activation: str
    ) -> Sequence[Tuple]:
        """Get the layout of the activations.

        Returns a list of tuple, describing each column as:
            - continuous, and with length 1 + number of GMM clusters.
            - discrete, and with length <N>, the length of the one-hot encoding.
        """
        out = []
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_type == "continuous":
                out.extend(
                    [
                        (continuous_activation, 1),
                        (
                            discrete_activation,
                            column_transform_info.output_dimensions - 1,
                        ),
                    ]
                )
            else:
                out.append(
                    (discrete_activation, column_transform_info.output_dimensions)
                )

        return out


class TimeSeriesTabularEncoder(TransformerMixin, BaseEstimator):
    """TimeSeries Tabular encoder.

    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        max_clusters: int = 10,
        categorical_limit: int = 10,
        whitelist: list = [],
    ) -> None:
        self.max_clusters = max_clusters
        self.categorical_limit = categorical_limit
        self.whitelist = whitelist

    def fit_temporal(
        self,
        temporal_data: List[pd.DataFrame],
        observation_times: List,
        discrete_columns: Optional[List] = None,
    ) -> "TimeSeriesTabularEncoder":
        # Temporal
        self.temporal_encoder = TabularEncoder(
            max_clusters=self.max_clusters,
            categorical_limit=self.categorical_limit,
            whitelist=self.whitelist,
        )
        temporal_features = temporal_data[0].columns

        temporal_arr = np.asarray(temporal_data)
        temporal_arr = np.swapaxes(temporal_arr, -1, 0).reshape(
            len(temporal_features), -1
        )
        temporal_df = pd.DataFrame(temporal_arr.T, columns=temporal_features)

        self.temporal_encoder.fit(temporal_df)

        # Temporal horizons
        self.observation_times_encoder = MinMaxScaler().fit(
            np.asarray(observation_times).reshape(-1, 1)
        )

        return self

    def fit(
        self,
        static_data: pd.DataFrame,
        temporal_data: List[pd.DataFrame],
        observation_times: List,
        discrete_columns: Optional[List] = None,
    ) -> "TimeSeriesTabularEncoder":
        # Static
        self.static_encoder = TabularEncoder(
            max_clusters=self.max_clusters,
            categorical_limit=self.categorical_limit,
            whitelist=self.whitelist,
        )
        self.static_encoder.fit(static_data, discrete_columns=discrete_columns)

        # Temporal
        self.fit_temporal(
            temporal_data, observation_times, discrete_columns=discrete_columns
        )

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def transform_observation_times(
        self,
        observation_times: List,
    ) -> List:
        horizons_encoded = (
            self.observation_times_encoder.transform(
                np.asarray(observation_times).reshape(-1, 1)
            )
            .reshape(len(observation_times), -1)
            .tolist()
        )
        return horizons_encoded

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def transform_temporal(
        self,
        temporal_data: List[pd.DataFrame],
        observation_times: List,
    ) -> Tuple[pd.DataFrame, List]:
        temporal_encoded = []
        for item in temporal_data:
            temporal_encoded.append(self.temporal_encoder.transform(item))

        horizons_encoded = self.transform_observation_times(observation_times)

        return temporal_encoded, horizons_encoded

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def transform_static(
        self,
        static_data: pd.DataFrame,
    ) -> pd.DataFrame:
        static_encoded = self.static_encoder.transform(static_data)

        return static_encoded

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def transform(
        self,
        static_data: pd.DataFrame,
        temporal_data: List[pd.DataFrame],
        observation_times: List,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
        static_encoded = self.transform_static(static_data)

        temporal_encoded, horizons_encoded = self.transform_temporal(
            temporal_data, observation_times
        )

        return static_encoded, temporal_encoded, horizons_encoded

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit_transform_temporal(
        self,
        temporal_data: List[pd.DataFrame],
        observation_times: List,
    ) -> Tuple[pd.DataFrame, List]:
        return self.fit_temporal(temporal_data, observation_times).transform_temporal(
            temporal_data, observation_times
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit_transform(
        self,
        static_data: pd.DataFrame,
        temporal_data: List[pd.DataFrame],
        observation_times: List,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
        return self.fit(static_data, temporal_data, observation_times).transform(
            static_data, temporal_data, observation_times
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def inverse_transform_observation_times(
        self,
        observation_times: List,
    ) -> pd.DataFrame:
        horizons_decoded = (
            self.observation_times_encoder.inverse_transform(
                np.asarray(observation_times).reshape(-1, 1)
            )
            .reshape(len(observation_times), -1)
            .tolist()
        )
        return horizons_decoded

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def inverse_transform_temporal(
        self,
        temporal_encoded: List[pd.DataFrame],
        observation_times: List,
    ) -> pd.DataFrame:
        temporal_decoded = []
        for item in temporal_encoded:
            temporal_decoded.append(self.temporal_encoder.inverse_transform(item))

        horizons_decoded = self.inverse_transform_observation_times(observation_times)

        return temporal_decoded, horizons_decoded

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def inverse_transform_static(
        self,
        static_encoded: pd.DataFrame,
    ) -> pd.DataFrame:
        static_decoded = self.static_encoder.inverse_transform(static_encoded)
        return static_decoded

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def inverse_transform(
        self,
        static_encoded: pd.DataFrame,
        temporal_encoded: List[pd.DataFrame],
        observation_times: List,
    ) -> pd.DataFrame:
        static_decoded = self.inverse_transform_static(static_encoded)

        temporal_decoded, horizons_decoded = self.inverse_transform_temporal(
            temporal_encoded, observation_times
        )
        return static_decoded, temporal_decoded, horizons_decoded

    def layout(self) -> Tuple[List, List]:
        return self.static_encoder.layout(), self.temporal_encoder.layout()

    def n_features(self) -> Tuple:
        return self.static_encoder.n_features(), self.temporal_encoder.n_features()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def activation_layout_temporal(
        self, discrete_activation: str, continuous_activation: str
    ) -> Any:
        return self.temporal_encoder.activation_layout(
            discrete_activation, continuous_activation
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def activation_layout(
        self, discrete_activation: str, continuous_activation: str
    ) -> Tuple:
        return self.static_encoder.activation_layout(
            discrete_activation, continuous_activation
        ), self.temporal_encoder.activation_layout(
            discrete_activation, continuous_activation
        )


class TimeSeriesBinEncoder(TransformerMixin, BaseEstimator):
    """Time series Bin encoder.

    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        max_clusters: int = 10,
        categorical_limit: int = 10,
    ) -> None:
        """Create a data transformer.

        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
        """
        self.encoder = BinEncoder(
            max_clusters=max_clusters,
            categorical_limit=categorical_limit,
        )

    def _prepare(
        self,
        static_data: pd.DataFrame,
        temporal_data: List[pd.DataFrame],
        observation_times: List,
    ) -> pd.DataFrame:
        temporal_init = np.asarray(temporal_data)[:, 0, :].squeeze()
        temporal_init_df = pd.DataFrame(temporal_init, columns=temporal_data[0].columns)

        out = pd.concat(
            [
                static_data.reset_index(drop=True),
                temporal_init_df.reset_index(drop=True),
            ],
            axis=1,
        )
        out.columns = np.asarray(range(len(out.columns)))
        out.columns = out.columns.astype(str)

        return out

    def fit(
        self,
        static_data: pd.DataFrame,
        temporal_data: List[pd.DataFrame],
        observation_times: List,
        discrete_columns: Optional[List] = None,
    ) -> "TimeSeriesBinEncoder":
        """Fit the TimeSeriesBinEncoder"""

        data = self._prepare(static_data, temporal_data, observation_times)

        self.encoder.fit(data, discrete_columns=discrete_columns)
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def transform(
        self,
        static_data: pd.DataFrame,
        temporal_data: List[pd.DataFrame],
        observation_times: List,
    ) -> pd.DataFrame:
        """Take raw data and output a matrix data."""
        data = self._prepare(static_data, temporal_data, observation_times)
        return self.encoder.transform(data)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit_transform(
        self,
        static: pd.DataFrame,
        temporal: List[pd.DataFrame],
        observation_times: List,
    ) -> pd.DataFrame:
        return self.fit(static, temporal, observation_times).transform(
            static, temporal, observation_times
        )
