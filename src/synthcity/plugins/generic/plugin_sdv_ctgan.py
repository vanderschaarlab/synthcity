# stdlib
from typing import Any, List

# third party
import pandas as pd
from ctgan import CTGANSynthesizer

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


class OriginalCTGANPlugin(Plugin):
    """Conditional tabular GAN implementation from the SDV package.

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
        >>> from sklearn.datasets import load_iris
        >>> from synthcity.plugins import Plugins
        >>>
        >>> X, y = load_iris(as_frame = True, return_X_y = True)
        >>> X["target"] = y
        >>>
        >>> plugin = Plugins().get("sdv_ctgan", n_iter = 100)
        >>> plugin.fit(X)
        >>>
        >>> plugin.generate(50)


    Reference: Xu, Lei, Maria Skoularidou, Alfredo Cuesta-Infante, and Kalyan Veeramachaneni. "Modeling tabular data using conditional gan." Advances in Neural Information Processing Systems 32 (2019).
    """

    def __init__(
        self,
        n_iter: int = 2000,
        embedding_n_units: int = 128,
        generator_n_units: int = 500,
        generator_n_layers: int = 3,
        generator_lr: float = 2e-4,
        generator_decay: float = 1e-6,
        discriminator_n_units: int = 500,
        discriminator_n_layers: int = 3,
        discriminator_lr: float = 2e-4,
        discriminator_decay: float = 1e-6,
        batch_size: int = 500,
        discriminator_steps: int = 1,
        pac: int = 10,
        cat_limit: int = 15,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.cat_limit = cat_limit
        self.model = CTGANSynthesizer(
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
        return "sdv_ctgan"

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

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "OriginalCTGANPlugin":
        discrete_columns = []

        for col in X.columns:
            if len(X[col].unique()) < self.cat_limit:
                discrete_columns.append(col)

        self.model.fit(X.dataframe(), discrete_columns=discrete_columns)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.sample, count, syn_schema)


plugin = OriginalCTGANPlugin
