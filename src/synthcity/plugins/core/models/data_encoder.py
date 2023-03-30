# stdlib
from typing import Any, List, Type, Union

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    QuantileTransformer,
    StandardScaler,
)

FeatureEncoder = Any


class FeatureEncoder(TransformerMixin, BaseEstimator):  # type: ignore
    """Base feature encoder, with sklearn-style API"""

    def __new__(cls, **kwargs: Any) -> FeatureEncoder:
        obj = super().__new__()
        obj.__dict__.update(kwargs)  # auto set all parameters as attributes
        return obj

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, x: pd.Series, y: Any = None, **kwargs: Any) -> FeatureEncoder:
        self.feature_name_in = x.name
        out = self._fit(x, **kwargs)._transform(x)

        if np.ndim(out) == 1:
            self.n_features_out = 1
        else:
            self.n_features_out = np.shape(out)[1]

        self.feature_names_out = self.get_feature_names_out()

        return self

    def _fit(self, x: pd.Series, **kwargs: Any) -> FeatureEncoder:
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def transform(self, x: pd.Series) -> Any:
        out = self._transform(x)
        if isinstance(out, np.ndarray):
            if out.ndim == 1:
                return pd.Series(out, self.feature_name_in)
            else:
                return pd.DataFrame(out, columns=self.feature_names_out)
        return out

    def _transform(self, x: pd.Series) -> Any:
        return x

    def get_feature_names_out(self) -> List[str]:
        n = self.n_features_out
        if n == 1:
            return [self.feature_name_in]
        else:
            return [self.feature_name_in + str(i) for i in range(n)]

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def inverse_transform(self, data: Any) -> pd.Series:
        x = self._inverse_transform(data)
        return pd.Series(x, name=self.feature_name_in)

    def _inverse_transform(self, data: Any) -> pd.Series:
        return data

    @classmethod
    def wraps(cls, encoder_class: TransformerMixin) -> Type[FeatureEncoder]:
        """Wraps sklearn transformer to FeatureEncoder."""

        class WrappedEncoder(FeatureEncoder):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.encoder = encoder_class(*args, **kwargs)

            def _fit(self, x: pd.Series, **kwargs: Any) -> FeatureEncoder:
                self.encoder.fit(x, **kwargs)
                return self

            def _transform(self, x: pd.Series) -> Any:
                return self.encoder.transform(x)

            def _inverse_transform(self, x: pd.Series) -> Any:
                return self.encoder.inverse_transform(x)

            def get_feature_names_out(self) -> List[str]:
                return self.encoder.get_feature_names_out([self.feature_name_in])

        for attr in (
            "__module__",
            "__name__",
            "__qualname__",
            "__doc__",
            "__annotations__",
        ):
            setattr(WrappedEncoder, attr, getattr(encoder_class, attr))

        return WrappedEncoder


OneHotEncoder = FeatureEncoder.wraps(OneHotEncoder)
StandardScaler = FeatureEncoder.wraps(StandardScaler)
MinMaxScaler = FeatureEncoder.wraps(MinMaxScaler)


class DatetimeEncoder(FeatureEncoder):
    """Datetime variables encoder"""

    def _transform(self, x: pd.Series) -> pd.Series:
        return pd.to_numeric(x).astype(float)

    def _inverse_transform(self, data: pd.Series) -> pd.Series:
        return pd.to_datetime(data)


class BayesianGMMEncoder(FeatureEncoder):
    """Bayesian Gaussian Mixture encoder"""

    def __init__(
        self,
        n_components: int = 10,
        random_state: int = 0,
        weight_threshold: float = 0.005,
        clip_output: bool = True,
        std_multiplier: int = 4,
    ) -> None:
        self.model = BayesianGaussianMixture(
            n_components=n_components,
            random_state=random_state,
            weight_concentration_prior=1e-3,
        )
        self.weights: List[float]

    def _fit(self, x: pd.Series, **kwargs: Any) -> "BayesianGaussianMixture":
        self.min_value = x.min()
        self.max_value = x.max()

        self.model.fit(x.values.reshape(-1, 1))
        self.weights = self.model.weights_
        self.means = self.model.means_.reshape(-1)
        self.stds = np.sqrt(self.model.covariances_).reshape(-1)

        return self

    def _transform(self, x: pd.Series) -> pd.DataFrame:
        x = x.values.reshape(-1, 1)
        means = self.means.reshape(1, -1)
        stds = self.stds.reshape(1, -1)

        # predict cluster value
        normalized_values = (x - means) / (self.std_multiplier * stds)

        # predict cluster
        component_probs = self.model.predict_proba(x)

        components = np.argmax(component_probs, axis=1)

        normalized = normalized_values[np.arange(len(x)), components]
        if self.clip_output:
            normalized = np.clip(normalized, -0.99, 0.99)
        normalized = normalized.reshape(-1, 1)

        components = np.eye(self.n_components)[components]  # onehot
        return np.hstack([normalized, components])

    def get_feature_names_out(self) -> List[str]:
        name = self.feature_name_in
        return [f"{name}.value"] + [
            f"{name}.component_{i}" for i in range(self.n_features_out - 1)
        ]

    def _inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        if self.clip_output:
            data = np.clip(data.values[:, 0], -1, 1)

        means = self.model.means_.reshape([-1])
        stds = np.sqrt(self.model.covariances_).reshape([-1])
        components = np.argmax(data.values[:, 1:], axis=1)

        # recreate data
        std_t = stds[components]
        mean_t = means[components]
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
    ):
        super().__init__(
            n_quantiles=None,
            output_distribution="normal",
            ignore_implicit_zeros=ignore_implicit_zeros,
            subsample=subsample,
            random_state=random_state,
            copy=copy,
        )

    def fit(self, x: pd.Series, y: Any = None) -> "GaussianQuantileTransformer":
        self.n_quantiles = max(min(len(x) // 30, 1000), 10)
        return super().fit(x, y)


ENCODERS = {
    "datetime": DatetimeEncoder,
    "onehot": OneHotEncoder,
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "quantile": GaussianQuantileTransformer,
    "bayesian_gmm": BayesianGMMEncoder,
}


def get_encoder(encoder: Union[str, type]) -> Type[FeatureEncoder]:
    """Get a registered encoder.

    Supported encoders:
    - Datetime
        - datetime
    - Categorical
        - onehot
    - Continuous
        - standard
        - minmax
        - quantile
        - bayesian_gmm
    """
    if isinstance(encoder, type):  # custom encoder
        return FeatureEncoder.wraps(encoder)
    return ENCODERS[encoder]
