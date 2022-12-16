# stdlib
from typing import Any, List

# third party
import pandas as pd
from ctgan import TVAESynthesizer

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    IntegerDistribution,
)
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema

pd.options.mode.chained_assignment = None


class OriginalTVAEPlugin(Plugin):
    """Tabular VAE implementation from the SDV package.

    Args:
        embedding_n_units: int = 128
            Size of the random sample passed to the Generator.
        compress_n_units: int = 128
            Size of each hidden layer in the encoder.
        compress_n_layers: int = 2
            Number of layers in the encoder.
        decompress_n_units: int = 128
            Size of each hidden layer in the decoder.
        decompress_n_layers: int = 2
            Number of layers in the decoder
        l2scale: float = 1e-5
            Regularization term.
        batch_size: int = 500
            Number of data samples to process in each step.
        n_iter: int = 300
            Number of training n_iter.
        loss_factor: int = 2
            Multiplier for the reconstruction error.

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from synthcity.plugins import Plugins
        >>>
        >>> X, y = load_iris(as_frame = True, return_X_y = True)
        >>> X["target"] = y
        >>>
        >>> plugin = Plugins().get("sdv_tvae", n_iter = 100)
        >>> plugin.fit(X)
        >>>
        >>> plugin.generate(50)


    Reference: Xu, Lei, Maria Skoularidou, Alfredo Cuesta-Infante, and Kalyan Veeramachaneni. "Modeling tabular data using conditional gan." Advances in Neural Information Processing Systems 32 (2019).

    """

    def __init__(
        self,
        embedding_n_units: int = 500,
        compress_n_units: int = 500,
        compress_n_layers: int = 3,
        decompress_n_units: int = 500,
        decompress_n_layers: int = 3,
        l2scale: float = 1e-5,
        batch_size: int = 500,
        n_iter: int = 2000,
        loss_factor: int = 1,
        cat_limit: int = 15,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.cat_limit = cat_limit
        self.model = TVAESynthesizer(
            embedding_dim=embedding_n_units,
            compress_dims=list(compress_n_units for i in range(compress_n_layers)),
            decompress_dims=list(
                decompress_n_units for i in range(decompress_n_layers)
            ),
            l2scale=l2scale,
            batch_size=batch_size,
            epochs=n_iter,
            loss_factor=loss_factor,
        )

    @staticmethod
    def name() -> str:
        return "sdv_tvae"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="embedding_n_units", low=100, high=500, step=50),
            IntegerDistribution(name="compress_n_units", low=100, high=500, step=50),
            IntegerDistribution(name="decompress_n_units", low=100, high=500, step=50),
            IntegerDistribution(name="compress_n_layers", low=1, high=3),
            IntegerDistribution(name="decompress_n_layers", low=1, high=3),
            IntegerDistribution(name="batch_size", low=100, high=300, step=50),
            IntegerDistribution(name="n_iter", low=100, high=500, step=50),
            CategoricalDistribution(name="l2scale", choices=[1e-5, 1e-4, 1e-3]),
            IntegerDistribution(name="loss_factor", low=1, high=5),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "OriginalTVAEPlugin":
        discrete_columns = []

        for col in X.columns:
            if len(X[col].unique()) < self.cat_limit:
                discrete_columns.append(col)

        self.model.fit(X.dataframe(), discrete_columns=discrete_columns)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.sample, count, syn_schema)


plugin = OriginalTVAEPlugin
