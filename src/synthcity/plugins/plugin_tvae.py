# stdlib
from typing import Any, List

# third party
import pandas as pd
from sdv.tabular import TVAE

# synthcity absolute
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    IntegerDistribution,
)
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class TVAEPlugin(Plugin):
    """TVAE plugin based on the VAE-based Deep Learning data synthesizer which was presented at the NeurIPS 2020 conference by the paper titled Modeling Tabular data using Conditional GAN.

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
        epochs: int = 300
            Number of training epochs.
        loss_factor: int = 2
            Multiplier for the reconstruction error.

    Example:
        >>> from synthcity.plugins import Plugins
        >>> plugin = Plugins().get("tvae")
        >>> from sklearn.datasets import load_iris
        >>> X = load_iris()
        >>> plugin.fit(X)
        >>> plugin.generate()
    """

    def __init__(
        self,
        embedding_n_units: int = 128,
        compress_n_units: int = 128,
        compress_n_layers: int = 2,
        decompress_n_units: int = 128,
        decompress_n_layers: int = 2,
        l2scale: float = 1e-5,
        batch_size: int = 500,
        epochs: int = 300,
        loss_factor: int = 2,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.model = TVAE(
            embedding_dim=embedding_n_units,
            compress_dims=list(compress_n_units for i in range(compress_n_layers)),
            decompress_dims=list(
                decompress_n_units for i in range(decompress_n_layers)
            ),
            l2scale=l2scale,
            batch_size=batch_size,
            epochs=epochs,
            loss_factor=loss_factor,
        )

    @staticmethod
    def name() -> str:
        return "tvae"

    @staticmethod
    def type() -> str:
        return "vae"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="embedding_n_units", low=100, high=500, step=50),
            IntegerDistribution(name="compress_n_units", low=100, high=500, step=50),
            IntegerDistribution(name="decompress_n_units", low=100, high=500, step=50),
            IntegerDistribution(name="compress_n_layers", low=1, high=3),
            IntegerDistribution(name="decompress_n_layers", low=1, high=3),
            IntegerDistribution(name="batch_size", low=100, high=300, step=50),
            IntegerDistribution(name="epochs", low=100, high=500, step=50),
            CategoricalDistribution(name="l2scale", choices=[1e-5, 1e-4, 1e-3]),
            IntegerDistribution(name="loss_factor", low=1, high=5),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "TVAEPlugin":
        self.model.fit(X)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.sample, count, syn_schema)


plugin = TVAEPlugin
