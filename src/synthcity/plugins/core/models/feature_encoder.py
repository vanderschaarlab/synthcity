# stdlib
from typing import Any, List, Optional, Type, Union

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


def validate_shape(x: np.ndarray, n_dim: int) -> np.ndarray:
    if n_dim == 1:
        if x.ndim == 2:
            x = np.squeeze(x, axis=1)
        if x.ndim != 1:
            raise ValueError("array must be 1D")
        return x
    elif n_dim == 2:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("array must be 2D")
        return x
    else:
        raise ValueError("n_dim must be 1 or 2")


FeatureEncoder = Any  # tried to use ForwardRef but it didn't work under mypy


class FeatureEncoder(TransformerMixin, BaseEstimator):  # type: ignore
    """
    Base feature encoder with sklearn-style API.
    """

    n_dim_in: int = 1
    n_dim_out: int = 2
    n_features_out: int
    feature_name_in: str
    feature_names_out: List[str]
    feature_types_out: List[str]
    categorical: bool = False  # used by get_feature_types_out

    def __init__(
        self, n_dim_in: Optional[int] = None, n_dim_out: Optional[int] = None
    ) -> None:
        super().__init__()
        if n_dim_in is not None:
            self.n_dim_in = n_dim_in
        if n_dim_out is not None:
            self.n_dim_out = n_dim_out

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, x: pd.Series, y: Any = None, **kwargs: Any) -> FeatureEncoder:
        self.feature_name_in = x.name
        self.feature_type_in = self._get_feature_type(x)
        input = validate_shape(x.values, self.n_dim_in)
        output = self._fit(input, **kwargs)._transform(input)
        self._out_shape = (-1, *output.shape[1:])  # for inverse_transform
        output = validate_shape(output, self.n_dim_out)
        if self.n_dim_out == 1:
            self.n_features_out = 1
        else:
            self.n_features_out = output.shape[1]
        self.feature_names_out = self.get_feature_names_out()
        self.feature_types_out = self.get_feature_types_out(output)
        return self

    def _fit(self, x: np.ndarray, **kwargs: Any) -> FeatureEncoder:
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def transform(self, x: pd.Series) -> Union[pd.DataFrame, pd.Series]:
        data = validate_shape(x.values, self.n_dim_in)
        out = self._transform(data)
        out = validate_shape(out, self.n_dim_out)
        if self.n_dim_out == 1:
            return pd.Series(out, name=self.feature_name_in)
        else:
            return pd.DataFrame(out, columns=self.feature_names_out)

    def _transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def get_feature_names_out(self) -> List[str]:
        n = self.n_features_out
        if n == 1:
            return [self.feature_name_in]
        else:
            return [f"{self.feature_name_in}_{i}" for i in range(n)]

    def get_feature_types_out(self, output: np.ndarray) -> List[str]:
        t = self._get_feature_type(output)
        return [t] * self.n_features_out

    def _get_feature_type(self, x: Any) -> str:
        if self.categorical:
            return "discrete"
        elif np.issubdtype(x.dtype, np.floating):
            return "continuous"
        elif np.issubdtype(x.dtype, np.datetime64):
            return "datetime"
        else:
            return "discrete"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def inverse_transform(self, df: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        y = df.values.reshape(self._out_shape)
        x = self._inverse_transform(y)
        x = validate_shape(x, 1)
        return pd.Series(x, name=self.feature_name_in)

    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data

    @classmethod
    def wraps(
        cls: type, encoder_class: TransformerMixin, **params: Any
    ) -> Type[FeatureEncoder]:
        """Wraps sklearn transformer to FeatureEncoder."""

        class WrappedEncoder(FeatureEncoder):
            n_dim_in = 2  # most sklearn transformers accept 2D input

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.encoder = encoder_class(*args, **kwargs)

            def _fit(self, x: np.ndarray, **kwargs: Any) -> FeatureEncoder:
                self.encoder.fit(x, **kwargs)
                return self

            def _transform(self, x: np.ndarray) -> np.ndarray:
                return self.encoder.transform(x)

            def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
                return self.encoder.inverse_transform(data)

            def get_feature_names_out(self) -> List[str]:
                return list(self.encoder.get_feature_names_out([self.feature_name_in]))

        for attr in ("__name__", "__qualname__", "__doc__"):
            setattr(WrappedEncoder, attr, getattr(encoder_class, attr))
        for attr, val in params.items():
            setattr(WrappedEncoder, attr, val)

        return WrappedEncoder


OneHotEncoder = FeatureEncoder.wraps(
    OneHotEncoder, categorical=True, handle_unknown="ignore"
)
OrdinalEncoder = FeatureEncoder.wraps(OrdinalEncoder, categorical=True)
LabelEncoder = FeatureEncoder.wraps(LabelEncoder, n_dim_out=1, categorical=True)
StandardScaler = FeatureEncoder.wraps(StandardScaler)
MinMaxScaler = FeatureEncoder.wraps(MinMaxScaler)
RobustScaler = FeatureEncoder.wraps(RobustScaler)


class DatetimeEncoder(FeatureEncoder):
    """Datetime variables encoder"""

    n_dim_out = 1

    def _transform(self, x: np.ndarray) -> np.ndarray:
        return pd.to_numeric(x).astype(float)

    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return pd.to_datetime(data)


class BayesianGMMEncoder(FeatureEncoder):
    """Bayesian Gaussian Mixture encoder"""

    n_dim_in = 2

    def __init__(
        self,
        n_components: int = 10,
        random_state: int = 0,
        weight_threshold: float = 0.005,
        clip_output: bool = True,
        std_multiplier: int = 4,
    ) -> None:
        self.n_components = n_components
        self.weight_threshold = weight_threshold
        self.clip_output = clip_output
        self.std_multiplier = std_multiplier
        self.model = BayesianGaussianMixture(
            n_components=n_components,
            random_state=random_state,
            weight_concentration_prior=1e-3,
        )

    def _fit(self, x: np.ndarray, **kwargs: Any) -> "BayesianGaussianMixture":
        self.min_value = x.min()
        self.max_value = x.max()

        self.model.fit(x)
        self.weights = self.model.weights_
        self.means = self.model.means_.reshape(-1)
        self.stds = np.sqrt(self.model.covariances_).reshape(-1)

        return self

    def _transform(self, x: np.ndarray) -> np.ndarray:
        means = self.means.reshape(1, -1)
        stds = self.stds.reshape(1, -1)

        # predict cluster value
        normalized_values = (x - means) / (self.std_multiplier * stds)

        # predict cluster
        component_probs = self.model.predict_proba(x)

        components = np.argmax(component_probs, axis=1)

        normalized = normalized_values[np.arange(len(x)), components]
        if self.clip_output:  # why use 0.99 instead of 1?
            normalized = np.clip(normalized, -0.99, 0.99)
        normalized = normalized.reshape(-1, 1)

        components = np.eye(self.n_components, dtype=int)[components]
        return np.hstack([normalized, components])

    def get_feature_names_out(self) -> List[str]:
        name = self.feature_name_in
        return [f"{name}.value"] + [
            f"{name}.component_{i}" for i in range(self.n_components)
        ]

    def get_feature_types_out(self, output: np.ndarray) -> List[str]:
        return ["continuous"] + ["discrete"] * self.n_components

    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        components = np.argmax(data[:, 1:], axis=1)

        data = data[:, 0]
        if self.clip_output:
            data = np.clip(data, -1.0, 1.0)

        # recreate data
        mean_t = self.means[components]
        std_t = self.stds[components]
        reversed_data = data * self.std_multiplier * std_t + mean_t

        # clip values
        return np.clip(reversed_data, self.min_value, self.max_value)


@FeatureEncoder.wraps
class GaussianQuantileTransformer(QuantileTransformer):
    """Quantile transformer with Gaussian distribution"""

    def __init__(
        self,
        *,
        ignore_implicit_zeros: bool = False,
        subsample: int = 10000,
        random_state: Any = None,
        copy: bool = True,
    ) -> None:
        super().__init__(
            n_quantiles=None,
            output_distribution="normal",
            ignore_implicit_zeros=ignore_implicit_zeros,
            subsample=subsample,
            random_state=random_state,
            copy=copy,
        )

    def fit(self, x: np.ndarray, y: Any = None) -> "GaussianQuantileTransformer":
        self.n_quantiles = max(min(len(x) // 30, 1000), 10)
        return super().fit(x, y)
