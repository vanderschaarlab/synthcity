# stdlib
from functools import wraps
from typing import Any, List, Optional, Union

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


class _DataEncoder(TransformerMixin, BaseEstimator):
    """Base data encoder, with sklearn-style API"""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: Any) -> Any:
        return self._fit(X)

    def _fit(self, X: Any) -> Any:
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def transform(self, X: Any) -> Any:
        return self._transform(X)

    def _transform(self, X: Any) -> Any:
        return X

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def inverse_transform(self, X: Any) -> Any:
        return self._inverse_transform(X)

    def _inverse_transform(self, X: Any) -> Any:
        return X

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit_transform(self, X: Any) -> Any:
        return self.fit(X).transform(X)

    @classmethod
    def wraps(cls, encoder_class: TransformerMixin) -> type:
        """Wraps sklearn encoder to DataEncoder."""

        @wraps(encoder_class)
        class WrappedEncoder(_DataEncoder):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.encoder = encoder_class(*args, **kwargs)

            def _fit(self, X: Any) -> _DataEncoder:
                self.encoder.fit(X)
                return self

            def _transform(self, X: Any) -> Any:
                return self.encoder.transform(X)

            def _inverse_transform(self, X: Any) -> Any:
                return self.encoder.inverse_transform(X)

        return WrappedEncoder


class DatetimeEncoder(_DataEncoder):
    """Datetime variables encoder"""

    def _transform(self, X: pd.Series) -> pd.Series:
        return pd.to_numeric(X).astype(float)

    def _inverse_transform(self, X: pd.Series) -> pd.Series:
        return pd.to_datetime(X)


class BayesianGMMEncoder(_DataEncoder):
    """Bayesian Gaussian Mixture encoder"""

    def __init__(
        self,
        n_components: int = 10,
        random_state: int = 0,
        weight_threshold: float = 0.005,
    ) -> None:
        self.model = BayesianGaussianMixture(
            n_components=n_components,
            random_state=random_state,
            weight_concentration_prior=1e-3,
        )
        self.n_components = n_components
        self.weight_threshold = weight_threshold
        self.weights: Optional[List[float]] = None
        self.std_multiplier = 4

    def _fit(self, X: pd.DataFrame) -> Any:
        self.min_value = X.min()
        self.max_value = X.max()

        self.model.fit(X.values.reshape(-1, 1))
        self.weights = self.model.weights_
        self.n_components = len(self.model.weights_)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        name = X.name
        X = X.values.reshape(-1, 1)
        means = self.model.means_.reshape(1, self.n_components)

        # predict cluster value
        stds = np.sqrt(self.model.covariances_).reshape(1, self.n_components)

        normalized_values = (X - means) / (self.std_multiplier * stds)

        # predict cluster
        component_probs = self.model.predict_proba(X)

        components = np.argmax(component_probs, axis=1)

        aranged = np.arange(len(X))
        normalized = normalized_values[aranged, components].reshape([-1, 1])
        normalized = np.clip(normalized, -0.99, 0.99).squeeze(axis=1)
        out = np.stack([normalized, components], axis=1)

        return pd.DataFrame(out, columns=[f"{name}.value", f"{name}.component"])

    def _inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        normalized = np.clip(X.values[:, 0], -1, 1)
        means = self.model.means_.reshape([-1])
        stds = np.sqrt(self.model.covariances_).reshape([-1])
        selected_component = X.values[:, 1].astype(int)

        # recreate data
        std_t = stds[selected_component]
        mean_t = means[selected_component]
        reversed_data = normalized * self.std_multiplier * std_t + mean_t

        # clip values
        return np.clip(reversed_data, self.min_value, self.max_value)


OneHotEncoder = _DataEncoder.wraps(OneHotEncoder)
StandardScaler = _DataEncoder.wraps(StandardScaler)
MinMaxScaler = _DataEncoder.wraps(MinMaxScaler)


@_DataEncoder.wraps
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

    def fit(self, X: pd.DataFrame, y: Any = None) -> "GaussianQuantileTransformer":
        self.n_quantiles = max(min(len(X) // 30, 1000), 10)
        return super().fit(X, y)


REGISTRY = {
    "datetime": DatetimeEncoder,
    "onehot": OneHotEncoder,
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "quantile": GaussianQuantileTransformer,
    "bayesian_gmm": BayesianGMMEncoder,
}


def get_encoder(encoder: Union[str, type]) -> TransformerMixin:
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
        return encoder
    return REGISTRY[encoder]
