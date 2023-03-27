"""TabularEncoder module.
"""

# stdlib
from typing import Any, List, Optional, Sequence, Tuple

# third party
import numpy as np
import pandas as pd
from pydantic import BaseModel, validate_arguments, validator
from sklearn.base import BaseEstimator, TransformerMixin

# synthcity absolute
import synthcity.logger as log
from synthcity.utils.dataframe import discrete_columns as find_cat_cols
from synthcity.utils.serialization import dataframe_hash

# synthcity relative
from .data_encoder import get_encoder


class FeatureInfo(BaseModel):
    name: str
    feature_type: str
    transform: Any
    output_dimensions: int
    transformed_features: List[str]

    @validator("feature_type")
    def _feature_type_validator(cls: Any, v: str) -> str:
        if v not in ["discrete", "continuous"]:
            raise ValueError(f"Invalid feature type {v}")
        return v

    @validator("transform")
    def _transform_validator(cls: Any, v: Any) -> Any:
        if not (
            hasattr(v, "fit")
            and hasattr(v, "transform")
            and hasattr(v, "inverse_transform")
        ):
            raise ValueError(f"Invalid transform {v}")
        return v

    @validator("output_dimensions")
    def _output_dimensions_validator(cls: Any, v: int) -> int:
        if v <= 0:
            raise ValueError(f"Invalid output_dimensions {v}")
        return v


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
        categorical_encoder: str = "onehot",
        continuous_encoder: str = "bayesian_gmm",
    ) -> None:
        """Create a data transformer.

        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
        """
        self.max_clusters = max_clusters
        self.categorical_limit = categorical_limit
        self.whitelist = whitelist
        self.categorical_encoder = categorical_encoder
        self.continuous_encoder = continuous_encoder

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _fit_continuous(self, data: pd.Series) -> FeatureInfo:
        """Fit the continuous encoder on a continuous column.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``FeatureInfo`` object.
        """
        name = data.name

        if self.continuous_encoder == "bayesian_gmm":
            encoder = get_encoder("bayesian_gmm")(
                n_components=min(self.max_clusters, len(data)),
            )
            n_components = encoder.n_components
            dim_out = 1 + n_components
            transformed_features = [f"{name}.value"] + [
                f"{name}.component_{i}" for i in range(n_components)
            ]
        else:
            encoder = get_encoder(self.continuous_encoder)()
            dim_out = 1
            transformed_features = [name]

        encoder.fit(data)

        return FeatureInfo(
            name=name,
            feature_type="continuous",
            transform=encoder,
            output_dimensions=dim_out,
            transformed_features=transformed_features,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _fit_discrete(self, data: pd.Series) -> FeatureInfo:
        """Fit one hot encoder for discrete column.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``FeatureInfo`` object.
        """
        name = data.name

        if self.categorical_encoder == "onehot":
            encoder = get_encoder("onehot")(handle_unknown="ignore", sparse=False)
        else:
            raise ValueError(f"Unknown categorical encoder {self.categorical_encoder}")

        encoder.fit(data.values.reshape(-1, 1))
        num_categories = len(encoder.categories_[0])

        transformed_features = list(encoder.get_feature_names_out([data.name]))

        return FeatureInfo(
            name=name,
            feature_type="discrete",
            transform=encoder,
            output_dimensions=num_categories,
            transformed_features=transformed_features,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self, raw_data: pd.DataFrame, discrete_columns: Optional[List] = None
    ) -> Any:
        """Fit the ``TabularEncoder``.

        This step also counts the #columns in matrix data and span information.
        """
        if discrete_columns is None:
            discrete_columns = find_cat_cols(raw_data, self.categorical_limit)

        self.output_dimensions = 0

        self._column_raw_dtypes = raw_data.infer_objects().dtypes
        self._column_transform_info = []

        for name in raw_data.columns:
            if name in self.whitelist:
                continue
            column_hash = dataframe_hash(raw_data[[name]])
            log.info(f"Encoding {name} {column_hash}")

            if name in discrete_columns:
                column_transform_info = self._fit_discrete(raw_data[name])
            else:
                column_transform_info = self._fit_continuous(raw_data[name])

            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info.append(column_transform_info)

        return self

    def _transform_continuous(
        self, column_transform_info: FeatureInfo, data: pd.Series
    ) -> pd.DataFrame:
        name = data.name
        encoder = column_transform_info.transform
        transformed = encoder.transform(data)

        #  Converts the transformed data to the appropriate output format.
        if self.continuous_encoder == "bayesian_gmm":
            output = np.zeros(
                (len(transformed), column_transform_info.output_dimensions)
            )
            output[:, 0] = transformed[f"{name}.value"].to_numpy()
            index = transformed[f"{name}.component"].to_numpy().astype(int)
            output[np.arange(index.size), index + 1] = 1
        else:
            output = transformed.to_numpy().reshape(-1, 1)

        return pd.DataFrame(
            output,
            columns=column_transform_info.transformed_features,
        )

    def _transform_discrete(
        self, column_transform_info: FeatureInfo, data: pd.Series
    ) -> pd.DataFrame:
        encoder = column_transform_info.transform
        return pd.DataFrame(
            encoder.transform(data.to_frame().values),
            columns=column_transform_info.transformed_features,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def transform(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Take raw data and output a matrix data."""
        if len(self._column_transform_info) == 0:
            return pd.DataFrame(np.zeros((len(raw_data), 0)))

        column_data_list = []
        for name in self.whitelist:
            if name not in raw_data.columns:
                continue
            data = raw_data[name]
            column_data_list.append(data)

        for column_transform_info in self._column_transform_info:
            name = column_transform_info.name
            data = raw_data[name]

            if column_transform_info.feature_type == "continuous":
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
        column_transform_info: FeatureInfo,
        column_data: pd.DataFrame,
    ) -> pd.DataFrame:
        encoder = column_transform_info.transform
        if self.continuous_encoder == "bayesian_gmm":
            data = pd.DataFrame(
                column_data.values[:, :2], columns=["value", "component"]
            )
            data.iloc[:, 1] = np.argmax(column_data.values[:, 1:], axis=1)
        else:
            data = column_data
        return encoder.inverse_transform(data)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _inverse_transform_discrete(
        self, column_transform_info: FeatureInfo, column_data: pd.DataFrame
    ) -> pd.DataFrame:
        encoder = column_transform_info.transform
        column = column_transform_info.name
        return pd.DataFrame(
            encoder.inverse_transform(column_data),
            columns=[column],
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        """
        if len(self._column_transform_info) == 0:
            return pd.DataFrame(np.zeros((len(data), 0)))

        st = 0
        recovered_column_data_list = []
        names = []
        feature_types = []

        for name in self.whitelist:
            if name not in data.columns:
                continue
            local_data = data[name]
            names.append(name)
            feature_types.append(self._column_raw_dtypes)
            recovered_column_data_list.append(local_data)

        for column_transform_info in self._column_transform_info:
            dim = column_transform_info.output_dimensions
            column_data = data.iloc[:, list(range(st, st + dim))]
            if column_transform_info.feature_type == "continuous":
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data
                )
            else:
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data
                )

            recovered_column_data_list.append(recovered_column_data)
            names.append(column_transform_info.name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = pd.DataFrame(
            recovered_data, columns=names, index=data.index
        ).astype(self._column_raw_dtypes.filter(names))
        return recovered_data

    def layout(self) -> List[Tuple]:
        """Get the layout of the encoded dataset.

        Returns a list of tuple, describing each column as:
            - continuous, and with length 1 + number of GMM clusters.
            - discrete, and with length <N>, the length of the one-hot encoding.
        """
        return self._column_transform_info

    def n_features(self) -> int:
        return np.sum(
            [
                column_transform_info.output_dimensions
                for column_transform_info in self._column_transform_info
            ]
        )

    def get_column_info(self, name: str) -> FeatureInfo:
        for column_transform_info in self._column_transform_info:
            if column_transform_info.name == name:
                return column_transform_info

        raise RuntimeError(f"Unknown column {name}")

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
        for column_transform_info in self._column_transform_info:
            if column_transform_info.feature_type == "continuous":
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


class BinEncoder(TabularEncoder):
    """Binary encoder (for SurvivalGAN).

    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    def _transform_continuous(
        self, column_transform_info: FeatureInfo, data: pd.Series
    ) -> pd.Series:
        name = data.name
        encoder = column_transform_info.transform
        transformed = encoder.transform(data)
        return transformed[f"{name}.component"].to_numpy().astype(int)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def transform(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Take raw data and output a matrix data."""
        output = raw_data.copy()

        for column_transform_info in self._column_transform_info:
            name = column_transform_info.name
            output[name] = self._transform_continuous(
                column_transform_info, raw_data[name]
            )

        return output


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
        encoder: str = "minmax",
    ) -> None:
        self.max_clusters = max_clusters
        self.categorical_limit = categorical_limit
        self.whitelist = whitelist
        self.encoder = encoder

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
        self.observation_times_encoder = get_encoder(self.encoder)
        self.observation_times_encoder.fit(np.asarray(observation_times).reshape(-1, 1))

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
        continuous_encoder: str = "gmm",
    ) -> None:
        """Create a data transformer.

        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
        """
        self.encoder = BinEncoder(
            max_clusters=max_clusters,
            categorical_limit=categorical_limit,
            continuous_encoder=continuous_encoder,
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
