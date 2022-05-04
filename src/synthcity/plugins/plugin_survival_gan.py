# stdlib
from typing import Any, List, Optional

# third party
import pandas as pd

# Necessary packages
from pydantic import validate_arguments
from torch.utils.data import sampler

# synthcity absolute
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.models import TabularGAN
from synthcity.plugins.models.time_to_event import select_uncensoring_model


class SurvivalGANPlugin(Plugin):
    """SurvivalGAN plugin.

    Args:
        generator_n_layers_hidden: int
            Number of hidden layers in the generator
        generator_n_units_hidden: int
            Number of hidden units in each layer of the Generator
        generator_nonlin: string, default 'tanh'
            Nonlinearity to use in the generator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        n_iter: int
            Maximum number of iterations in the Generator.
        generator_dropout: float
            Dropout value. If 0, the dropout is not used.
        discriminator_n_layers_hidden: int
            Number of hidden layers in the discriminator
        discriminator_n_units_hidden: int
            Number of hidden units in each layer of the discriminator
        discriminator_nonlin: string, default 'leaky_relu'
            Nonlinearity to use in the discriminator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        discriminator_n_iter: int
            Maximum number of iterations in the discriminator.
        discriminator_dropout: float
            Dropout value for the discriminator. If 0, the dropout is not used.
        lr: float
            learning rate for optimizer. step_size equivalent in the JAX version.
        weight_decay: float
            l2 (ridge) penalty for the weights.
        batch_size: int
            Batch size
        seed: int
            Seed used
        clipping_value: int, default 0
            Gradients clipping value. Zero disables the feature
        encoder_max_clusters: int
            The max number of clusters to create for continuous columns when encoding

    Example:
        >>> from synthcity.plugins import Plugins
        >>> X = load_rossi()
        >>> plugin = Plugins().get("survival_gan", target_column = "arrest", time_to_event_column="week")
        >>> plugin.fit(X)
        >>> plugin.generate()
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        target_column: str = "event",
        time_to_event_column: str = "duration",
        time_horizons: Optional[List] = None,
        n_iter: int = 1000,
        generator_n_layers_hidden: int = 1,
        generator_n_units_hidden: int = 150,
        generator_nonlin: str = "tanh",
        generator_dropout: float = 0.1,
        discriminator_n_layers_hidden: int = 2,
        discriminator_n_units_hidden: int = 100,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 100,
        seed: int = 0,
        clipping_value: int = 0,
        lambda_gradient_penalty: float = 10,
        lambda_identifiability_penalty: float = 0.1,
        encoder_max_clusters: int = 5,
        encoder: Any = None,
        dataloader_sampler: Optional[sampler.Sampler] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.target_column = target_column
        self.time_to_event_column = time_to_event_column
        self.time_horizons = time_horizons

        self.generator_n_layers_hidden = generator_n_layers_hidden
        self.generator_n_units_hidden = generator_n_units_hidden
        self.generator_nonlin = generator_nonlin
        self.n_iter = n_iter
        self.generator_dropout = generator_dropout

        self.discriminator_n_layers_hidden = discriminator_n_layers_hidden
        self.discriminator_n_units_hidden = discriminator_n_units_hidden
        self.discriminator_nonlin = discriminator_nonlin
        self.discriminator_n_iter = discriminator_n_iter
        self.discriminator_dropout = discriminator_dropout

        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.seed = seed
        self.clipping_value = clipping_value
        self.lambda_gradient_penalty = lambda_gradient_penalty

        self.encoder_max_clusters = encoder_max_clusters
        self.encoder = encoder
        self.dataloader_sampler = dataloader_sampler

    @staticmethod
    def name() -> str:
        return "survival_gan"

    @staticmethod
    def type() -> str:
        return "survival_analysis"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="generator_n_layers_hidden", low=1, high=4),
            IntegerDistribution(
                name="generator_n_units_hidden", low=50, high=150, step=50
            ),
            CategoricalDistribution(
                name="generator_nonlin", choices=["relu", "leaky_relu", "tanh", "elu"]
            ),
            IntegerDistribution(name="n_iter", low=100, high=1000, step=100),
            FloatDistribution(name="generator_dropout", low=0, high=0.2),
            IntegerDistribution(name="discriminator_n_layers_hidden", low=1, high=4),
            IntegerDistribution(
                name="discriminator_n_units_hidden", low=50, high=150, step=50
            ),
            CategoricalDistribution(
                name="discriminator_nonlin",
                choices=["relu", "leaky_relu", "tanh", "elu"],
            ),
            IntegerDistribution(name="discriminator_n_iter", low=1, high=5),
            FloatDistribution(name="discriminator_dropout", low=0, high=0.2),
            CategoricalDistribution(name="lr", choices=[1e-3, 2e-4, 1e-4]),
            CategoricalDistribution(name="weight_decay", choices=[1e-3, 1e-4]),
            CategoricalDistribution(name="batch_size", choices=[100, 200, 500]),
            IntegerDistribution(name="encoder_max_clusters", low=2, high=20),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "SurvivalGANPlugin":
        if self.target_column not in X.columns:
            raise ValueError(
                f"Event column {self.target_column} not found in the dataframe"
            )

        if self.time_to_event_column not in X.columns:
            raise ValueError(
                f"Time to event column {self.time_to_event_column} not found in the dataframe"
            )

        Xcov = X.drop(columns=[self.target_column, self.time_to_event_column])
        T = X[self.time_to_event_column]
        E = X[self.target_column]

        # Uncensoring
        self.uncensoring_model = select_uncensoring_model(Xcov, T, E)

        self.uncensoring_model.fit(Xcov, T, E)
        T_uncensored = pd.Series(self.uncensoring_model.predict(Xcov), index=Xcov.index)
        T_uncensored[E == 1] = T[E == 1]

        df_uncensored = Xcov
        df_uncensored[self.time_to_event_column] = T_uncensored

        features = df_uncensored.shape[1]

        # Synthetic data generator
        self.model = TabularGAN(
            df_uncensored,
            n_units_latent=features,
            batch_size=self.batch_size,
            generator_n_layers_hidden=self.generator_n_layers_hidden,
            generator_n_units_hidden=self.generator_n_units_hidden,
            generator_nonlin=self.generator_nonlin,
            generator_nonlin_out_discrete="softmax",
            generator_nonlin_out_continuous="tanh",
            generator_lr=self.lr,
            generator_residual=True,
            generator_n_iter=self.n_iter,
            generator_batch_norm=False,
            generator_dropout=self.generator_dropout,
            generator_weight_decay=self.weight_decay,
            discriminator_n_units_hidden=self.discriminator_n_units_hidden,
            discriminator_n_layers_hidden=self.discriminator_n_layers_hidden,
            discriminator_n_iter=self.discriminator_n_iter,
            discriminator_nonlin=self.discriminator_nonlin,
            discriminator_batch_norm=False,
            discriminator_dropout=self.discriminator_dropout,
            discriminator_lr=self.lr,
            discriminator_weight_decay=self.weight_decay,
            encoder=self.encoder,
            clipping_value=self.clipping_value,
            lambda_gradient_penalty=self.lambda_gradient_penalty,
            encoder_max_clusters=self.encoder_max_clusters,
            dataloader_sampler=self.dataloader_sampler,
        )
        self.model.fit(df_uncensored)

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _generate(count: int) -> pd.DataFrame:
            generated = self.model.generate(count)
            generated[
                self.target_column
            ] = 1  # everybody is uncensored in the synthetic data
            return generated

        return self._safe_generate(_generate, count, syn_schema)


plugin = SurvivalGANPlugin
