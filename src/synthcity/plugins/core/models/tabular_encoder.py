"""TabularEncoder module.
"""

# stdlib
from typing import Any, List, Optional, Sequence, Tuple, Union

# third party
import numpy as np
import pandas as pd
from pydantic import BaseModel, validate_arguments, validator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

# synthcity absolute
import synthcity.logger as log
from synthcity.utils.dataframe import discrete_columns as find_cat_cols
from synthcity.utils.serialization import dataframe_hash

# synthcity relative
from .factory import get_feature_encoder


class FeatureInfo(BaseModel):
    name: str
    feature_type: str
    transform: Any
    output_dimensions: int
    transformed_features: List[str]
    trans_feature_types: List[str]

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

    categorical_encoder: Union[str, type] = "onehot"
    continuous_encoder: Union[str, type] = "bayesian_gmm"
    cat_encoder_params: dict = dict(handle_unknown="ignore", sparse=False)
    cont_encoder_params: dict = dict(n_components=10)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        *,
        whitelist: tuple = (),
        max_clusters: int = 10,
        categorical_limit: int = 10,
        categorical_encoder: Optional[Union[str, type]] = None,
        continuous_encoder: Optional[Union[str, type]] = None,
        cat_encoder_params: Optional[dict] = None,
        cont_encoder_params: Optional[dict] = None,
    ) -> None:
        """Create a data transformer.

        Args:
            whitelist (tuple):
                Columns that will not be transformed.
        """
        self.whitelist = whitelist
        self.categorical_limit = categorical_limit
        self.max_clusters = max_clusters
        if categorical_encoder is not None:
            self.categorical_encoder = categorical_encoder
        if continuous_encoder is not None:
            self.continuous_encoder = continuous_encoder
        if cat_encoder_params is not None:
            self.cat_encoder_params = cat_encoder_params
        else:
            self.cat_encoder_params = self.cat_encoder_params.copy()
        if cont_encoder_params is not None:
            self.cont_encoder_params = cont_encoder_params
        else:
            self.cont_encoder_params = self.cont_encoder_params.copy()
        if self.continuous_encoder == "bayesian_gmm":
            self.cont_encoder_params["n_components"] = max_clusters

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _fit_feature(self, feature: pd.Series, feature_type: str) -> FeatureInfo:
        """Fit the feature encoder on a column.

        Args:
            feature (pd.Series):
                A column of a dataframe.
            feature_type (str):
                Type of the feature ('discrete' or 'continuous').

        Returns:
            FeatureInfo:
                Information of the fitted feature encoder.
        """
        if feature_type == "discrete":
            encoder = get_feature_encoder(
                self.categorical_encoder, self.cat_encoder_params
            )
        else:
            encoder = get_feature_encoder(
                self.continuous_encoder, self.cont_encoder_params
            )

        encoder.fit(feature)

        return FeatureInfo(
            name=feature.name,
            feature_type=feature_type,
            transform=encoder,
            output_dimensions=encoder.n_features_out,
            transformed_features=encoder.feature_names_out,
            trans_feature_types=encoder.feature_types_out,
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
        self._column_transform_info_list: Sequence[FeatureInfo] = []

        for name in raw_data.columns:
            if name in self.whitelist:
                continue
            column_hash = dataframe_hash(raw_data[[name]])
            log.info(f"Encoding {name} {column_hash}")
            ftype = "discrete" if name in discrete_columns else "continuous"
            column_transform_info = self._fit_feature(raw_data[name], ftype)

            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

        return self

    def _transform_feature(
        self, column_transform_info: FeatureInfo, feature: pd.Series
    ) -> pd.DataFrame:
        encoder = column_transform_info.transform
        return pd.DataFrame(
            encoder.transform(feature).values,
            columns=column_transform_info.transformed_features,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def transform(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Take raw data and output a matrix data."""
        if len(self._column_transform_info_list) == 0:
            return pd.DataFrame(np.zeros((len(raw_data), 0)))

        column_data_list = []
        for name in self.whitelist:
            if name not in raw_data.columns:
                continue
            feature = raw_data[name]
            column_data_list.append(feature)

        for column_transform_info in self._column_transform_info_list:
            feature = raw_data[column_transform_info.name]
            column_data_list.append(
                self._transform_feature(column_transform_info, feature)
            )

        result = pd.concat(column_data_list, axis=1)
        result.index = raw_data.index

        return result

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _inverse_transform_feature(
        self,
        column_transform_info: FeatureInfo,
        column_data: pd.DataFrame,
    ) -> pd.Series:
        encoder = column_transform_info.transform
        return encoder.inverse_transform(column_data)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        """
        if len(self._column_transform_info_list) == 0:
            return pd.DataFrame(np.zeros((len(data), 0)))

        st = 0
        names = []
        feature_types = []
        recovered_feature_list = []

        for name in self.whitelist:
            if name not in data.columns:
                continue
            names.append(name)
            feature_types.append(self._column_raw_dtypes)
            recovered_feature_list.append(data[name])

        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data.iloc[:, list(range(st, st + dim))]
            recovered_feature = self._inverse_transform_feature(
                column_transform_info, column_data
            )
            recovered_feature_list.append(recovered_feature)
            names.append(column_transform_info.name)
            st += dim

        recovered_data = np.column_stack(recovered_feature_list)
        recovered_data = pd.DataFrame(
            recovered_data, columns=names, index=data.index
        ).astype(self._column_raw_dtypes.filter(names))
        return recovered_data

    def layout(self) -> Sequence[FeatureInfo]:
        """Get the layout of the encoded dataset.

        Returns a list of tuple, describing each column as:
            - continuous, and with length 1 + number of GMM clusters.
            - discrete, and with length <N>, the length of the one-hot encoding.
        """
        return self._column_transform_info_list

    def n_features(self) -> int:
        return np.sum(
            column_transform_info.output_dimensions
            for column_transform_info in self._column_transform_info_list
        )

    def get_column_info(self, name: str) -> FeatureInfo:
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.name == name:
                return column_transform_info

        raise RuntimeError(f"Unknown column {name}")

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def activation_layout(
        self, discrete_activation: str, continuous_activation: str
    ) -> Sequence[Tuple[str, int]]:
        """Get the layout of the activations.

        Returns a list of tuple, describing each column as:
            - continuous, and with length 1 + number of GMM clusters.
            - discrete, and with length <N>, the length of the one-hot encoding.
        """
        out = []
        acts = dict(discrete=discrete_activation, continuous=continuous_activation)
        for column_transform_info in self._column_transform_info_list:
            ct = column_transform_info.trans_feature_types[0]
            d = 0
            for t in column_transform_info.trans_feature_types:
                if t != ct:
                    out.append((acts[ct], d))
                    ct = t
                    d = 0
                d += 1
            out.append((acts[ct], d))
        return out


class BinEncoder(TabularEncoder):
    """Binary encoder (for SurvivalGAN).

    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    continuous_encoder = "bayesian_gmm"
    cont_encoder_params = dict(n_components=2)
    categorical_encoder = "passthrough"  # "onehot"
    cat_encoder_params = dict()  # dict(handle_unknown="ignore", sparse=False)

    def _transform_feature(
        self, column_transform_info: FeatureInfo, feature: pd.Series
    ) -> pd.DataFrame:
        if column_transform_info.feature_type == "discrete":
            return super()._transform_feature(column_transform_info, feature)
        bgm = column_transform_info.transform
        out = bgm.transform(feature)
        return pd.DataFrame(
            out.values[:, 1:].argmax(axis=1), columns=[bgm.feature_name_in]
        )

    def _inverse_transform_feature(
        self, column_transform_info: FeatureInfo, column_data: pd.DataFrame
    ) -> pd.Series:
        if column_transform_info == "discrete":
            return super()._inverse_transform_feature(
                column_transform_info, column_data
            )
        bgm = column_transform_info.transform
        components = column_data.values.reshape(-1)
        features = bgm.means[components]
        return pd.Series(features, name=bgm.feature_name_in)


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
