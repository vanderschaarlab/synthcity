# stdlib
from typing import Any, List

# third party
import pandas as pd
from sdv.tabular import CTGAN

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    IntegerDistribution,
)
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class CTGANPlugin(Plugin):
    """CTGAN plugin.

    CTGAN model is based on the GAN-based Deep Learning data synthesizer which was presented at the NeurIPS 2020 conference by the paper titled Modeling Tabular data using Conditional GAN.

    Args:
        embedding_n_units: int
            Size of the random sample passed to the Generator.
        generator_n_units: int
            The size of a Residual layer
        generator_n_layers: int
            The number of residual layers
        generator_lr: float
            Learning rate for the generator.
        generator_decay: float
            Generator weight decay for the Adam Optimizer.
        discriminator_n_units: int
            Size of the output samples for each one of the Discriminator Layers.
        discriminator_n_layers: int
            Number of Discriminator layers.
        discriminator_lr: float
            Learning rate for the discriminator.
        discriminator_decay: float
            Discriminator weight decay for the Adam Optimizer.
        batch_size: int
            Number of data samples to process in each step. Must be multiple of <pac>.
        discriminator_steps: int
            Number of discriminator updates to do for each generator update.
        n_iter: int
            Number of training n_iter.
        pac: int
            Number of samples to group together when applying the discriminator.

    Example:
        >>> from synthcity.plugins import Plugins
        >>> plugin = Plugins().get("ctgan")
        >>> from sklearn.datasets import load_iris
        >>> X = load_iris()
        >>> plugin.fit(X)
        >>> plugin.generate()
    """

    def __init__(
        self,
        embedding_n_units: int = 128,
        generator_n_units: int = 256,
        generator_n_layers: int = 2,
        generator_lr: float = 2e-4,
        generator_decay: float = 1e-6,
        discriminator_n_units: int = 256,
        discriminator_n_layers: int = 2,
        discriminator_lr: float = 2e-4,
        discriminator_decay: float = 1e-6,
        batch_size: int = 500,
        discriminator_steps: int = 1,
        n_iter: int = 300,
        pac: int = 10,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.model = CTGAN(
            embedding_dim=embedding_n_units,
            generator_dim=tuple(generator_n_units for i in range(generator_n_layers)),
            generator_lr=generator_lr,
            generator_decay=generator_decay,
            discriminator_dim=tuple(
                discriminator_n_units for i in range(discriminator_n_layers)
            ),
            discriminator_lr=discriminator_lr,
            discriminator_decay=discriminator_decay,
            batch_size=batch_size,
            discriminator_steps=discriminator_steps,
            epochs=n_iter,
            pac=pac,
            verbose=False,
        )

    @staticmethod
    def name() -> str:
        return "ctgan"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="embedding_n_units", low=100, high=500, step=50),
            IntegerDistribution(name="generator_n_units", low=100, high=500, step=50),
            IntegerDistribution(
                name="discriminator_n_units", low=100, high=500, step=50
            ),
            IntegerDistribution(name="generator_n_layers", low=1, high=3),
            IntegerDistribution(name="discriminator_n_layers", low=1, high=3),
            IntegerDistribution(name="batch_size", low=100, high=300, step=50),
            IntegerDistribution(name="discriminator_steps", low=1, high=10),
            IntegerDistribution(name="n_iter", low=100, high=500, step=50),
            CategoricalDistribution(name="generator_lr", choices=[1e-3, 2e-4]),
            CategoricalDistribution(name="discriminator_lr", choices=[1e-3, 2e-4]),
            CategoricalDistribution(name="generator_decay", choices=[1e-3, 1e-6]),
            CategoricalDistribution(name="discriminator_decay", choices=[1e-3, 1e-6]),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "CTGANPlugin":
        self.model.fit(X.dataframe())
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.sample, count, syn_schema)


plugin = CTGANPlugin
