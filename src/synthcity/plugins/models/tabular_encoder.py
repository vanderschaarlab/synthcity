"""TabularEncoder module.

Adapted from https://github.com/sdv-dev/CTGAN
"""

# stdlib
from collections import namedtuple
from typing import List, Optional, Sequence, Tuple

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from rdt.transformers import BayesGMMTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

ColumnTransformInfo = namedtuple(
    "ColumnTransformInfo",
    ["column_name", "column_type", "transform", "output_dimensions"],
)


class TabularEncoder(TransformerMixin, BaseEstimator):
    """Tabular encoder.

    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(self, max_clusters: int = 10, weight_threshold: float = 0.005) -> None:
        """Create a data transformer.

        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
            weight_threshold (float):
                Weight threshold for a Gaussian distribution to be kept.
        """
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _fit_continuous(self, data: pd.DataFrame) -> ColumnTransformInfo:
        """Train Bayesian GMM for continuous columns.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        gm = BayesGMMTransformer(
            max_clusters=self._max_clusters, weight_threshold=self._weight_threshold
        )
        gm.fit(data, [column_name])
        num_components = sum(gm.valid_component_indicator)

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="continuous",
            transform=gm,
            output_dimensions=1 + num_components,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _fit_discrete(self, data: pd.DataFrame) -> ColumnTransformInfo:
        """Fit one hot encoder for discrete column.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        ohe.fit(data[column_name].values.reshape(-1, 1))
        num_categories = len(ohe.categories_[0])

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="discrete",
            transform=ohe,
            output_dimensions=num_categories,
        )

    def fit(
        self, raw_data: pd.DataFrame, discrete_columns: Optional[List] = None
    ) -> "TabularEncoder":
        """Fit the ``DataTransformer``.

        Fits a ``BayesGMMTransformer`` for continuous columns and a
        ``OneHotEncoder`` for discrete columns.

        This step also counts the #columns in matrix data and span information.
        """
        if discrete_columns is None:
            discrete_columns = []

            for col in raw_data.columns:
                if len(raw_data[col].unique()) < 15:
                    discrete_columns.append(col)

        self.output_dimensions = 0

        self._column_raw_dtypes = raw_data.infer_objects().dtypes
        self._column_transform_info_list = []
        for column_name in raw_data.columns:
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(raw_data[[column_name]])
            else:
                column_transform_info = self._fit_continuous(raw_data[[column_name]])

            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

        return self

    def _transform_continuous(
        self, column_transform_info: ColumnTransformInfo, data: pd.DataFrame
    ) -> pd.DataFrame:
        column_name = data.columns[0]
        gm = column_transform_info.transform
        transformed = gm.transform(data, [column_name])

        #  Converts the transformed data to the appropriate output format.
        #  The first column (ending in '.normalized') stays the same,
        #  but the lable encoded column (ending in '.component') is one hot encoded.
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed[f"{column_name}.normalized"].to_numpy()
        index = transformed[f"{column_name}.component"].to_numpy().astype(int)
        output[np.arange(index.size), index + 1] = 1.0

        return pd.DataFrame(
            output,
            columns=[f"{column_name}.normalized"]
            + [
                f"{column_name}.component_{i}"
                for i in range(column_transform_info.output_dimensions - 1)
            ],
        )

    def _transform_discrete(
        self, column_transform_info: ColumnTransformInfo, data: pd.DataFrame
    ) -> pd.DataFrame:
        ohe = column_transform_info.transform
        return pd.DataFrame(
            ohe.transform(data), columns=ohe.get_feature_names_out(data.columns)
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def transform(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Take raw data and output a matrix data."""
        column_data_list = []
        for column_transform_info in self._column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
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
            column_data.values[:, :2], columns=list(gm.get_output_types())
        )
        data.iloc[:, 1] = np.argmax(column_data.values[:, 1:], axis=1)

        return gm.reverse_transform(data, [column_transform_info.column_name])

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _inverse_transform_discrete(
        self, column_transform_info: ColumnTransformInfo, column_data: pd.DataFrame
    ) -> pd.DataFrame:
        ohe = column_transform_info.transform
        column = column_transform_info.column_name
        data = pd.DataFrame(column_data, columns=ohe.get_feature_names_out([column]))
        return pd.DataFrame(
            ohe.inverse_transform(data),
            columns=[column],
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
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
        ).astype(self._column_raw_dtypes)
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
                out.append(
                    (continuous_activation, column_transform_info.output_dimensions)
                )
            else:
                out.append(
                    (discrete_activation, column_transform_info.output_dimensions)
                )

        return out
