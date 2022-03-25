# stdlib
from typing import Any, List

# third party
import pandas as pd
from sdv.tabular import CopulaGAN

# synthcity absolute
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    IntegerDistribution,
)
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class CopulaGANPlugin(Plugin):
    """CopulaGAN plugin, based on a combination of GaussianCopula transformation and GANs.

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
        epochs: int
            Number of training epochs.

    Example:
        >>> from synthcity.plugins import Plugins
        >>> plugin = Plugins().get("copulagan")
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
        batch_size: int = 100,
        discriminator_steps: int = 1,
        epochs: int = 300,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.model = CopulaGAN()

    @staticmethod
    def name() -> str:
        return "copulagan"

    @staticmethod
    def type() -> str:
        return "gan"

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
            IntegerDistribution(name="epochs", low=100, high=500, step=50),
            CategoricalDistribution(name="generator_lr", choices=[1e-3, 2e-4]),
            CategoricalDistribution(name="discriminator_lr", choices=[1e-3, 2e-4]),
            CategoricalDistribution(name="generator_decay", choices=[1e-3, 1e-6]),
            CategoricalDistribution(name="discriminator_decay", choices=[1e-3, 1e-6]),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "CopulaGANPlugin":
        self.model.fit(X)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        constraints = syn_schema.as_constraints()

        data_synth = pd.DataFrame([], columns=self.schema().features())
        for it in range(self.sampling_patience):
            iter_samples = self.model.sample(count)
            iter_samples_df = pd.DataFrame(
                iter_samples, columns=self.schema().features()
            )
            iter_samples_df = syn_schema.adapt_dtypes(iter_samples_df)

            iter_synth_valid = constraints.match(iter_samples_df)
            data_synth = pd.concat([data_synth, iter_synth_valid], ignore_index=True)

            if len(data_synth) >= count:
                break

        data_synth = syn_schema.adapt_dtypes(data_synth).head(count)

        return data_synth


plugin = CopulaGANPlugin
