# stdlib
from typing import Any, Callable, List, Optional

# third party
import pandas as pd

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.models.tabular_encoder import TimeSeriesBinEncoder
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.time_series.plugin_timegan import TimeGANPlugin
from synthcity.utils.constants import DEVICE


class ConditionalTimeGANPlugin(Plugin):
    """Synthetic time series generation using conditional TimeGAN.

    Args:
        n_iter: int
            Maximum number of iterations in the Generator.
        n_units_in: int
            Number of features
        generator_n_layers_hidden: int
            Number of hidden layers in the generator
        generator_n_units_hidden: int
            Number of hidden units in each layer of the Generator
        generator_nonlin: string, default 'elu'
            Nonlinearity to use in the generator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        generator_batch_norm: bool
            Enable/disable batch norm for the generator
        generator_dropout: float
            Dropout value. If 0, the dropout is not used.
        generator_residual: bool
            Use residuals for the generator
        discriminator_n_layers_hidden: int
            Number of hidden layers in the discriminator
        discriminator_n_units_hidden: int
            Number of hidden units in each layer of the discriminator
        discriminator_nonlin: string, default 'relu'
            Nonlinearity to use in the discriminator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        discriminator_n_iter: int
            Maximum number of iterations in the discriminator.
        discriminator_batch_norm: bool
            Enable/disable batch norm for the discriminator
        discriminator_dropout: float
            Dropout value for the discriminator. If 0, the dropout is not used.
        lr: float
            learning rate for optimizer. step_size equivalent in the JAX version.
        weight_decay: float
            l2 (ridge) penalty for the weights.
        batch_size: int
            Batch size
        n_iter_print: int
            Number of iterations after which to print updates and check the validation loss.
        random_state: int
            random_state used
        clipping_value: int, default 0
            Gradients clipping value
        encoder_max_clusters: int
            The max number of clusters to create for continuous columns when encoding
        encoder:
            Pre-trained tabular encoder. If None, a new encoder is trained.
        device:
            Device to use for computation
        gamma_penalty
            Latent representation penalty
        moments_penalty: float = 100
            Moments(var and mean) penalty
        embedding_penalty: float = 10
            Embedding representation penalty

    Example:
        >>> from synthcity.plugins import Plugins
        >>> from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
        >>> from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
        >>> static, temporal, horizons, outcome = GoogleStocksDataloader().load()
        >>> loader = TimeSeriesDataLoader(
        >>>             temporal_data=temporal,
        >>>             temporal_horizons=horizons,
        >>>             static_data=static,
        >>>             outcome=outcome,
        >>> )
        >>>
        >>> plugin = Plugins().get("ctimegan", n_iter = 50)
        >>> plugin.fit(loader)
        >>>
        >>> plugin.generate(count = 10)
    """

    def __init__(
        self,
        n_iter: int = 1000,
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 150,
        generator_nonlin: str = "leaky_relu",
        generator_nonlin_out_discrete: str = "softmax",
        generator_nonlin_out_continuous: str = "tanh",
        generator_batch_norm: bool = False,
        generator_dropout: float = 0.01,
        generator_loss: Optional[Callable] = None,
        generator_lr: float = 1e-3,
        generator_weight_decay: float = 1e-3,
        generator_residual: bool = True,
        discriminator_n_layers_hidden: int = 3,
        discriminator_n_units_hidden: int = 300,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_batch_norm: bool = False,
        discriminator_dropout: float = 0.1,
        discriminator_loss: Optional[Callable] = None,
        discriminator_lr: float = 1e-3,
        discriminator_weight_decay: float = 1e-3,
        batch_size: int = 64,
        n_iter_print: int = 10,
        random_state: int = 0,
        clipping_value: int = 0,
        encoder_max_clusters: int = 20,
        encoder: Any = None,
        device: Any = DEVICE,
        mode: str = "RNN",
        gamma_penalty: float = 1,
        moments_penalty: float = 100,
        embedding_penalty: float = 10,
        **kwargs: Any
    ) -> None:
        super().__init__()

        self.n_iter = n_iter
        self.generator_n_layers_hidden = generator_n_layers_hidden
        self.generator_n_units_hidden = generator_n_units_hidden
        self.generator_nonlin = generator_nonlin
        self.generator_nonlin_out_discrete = generator_nonlin_out_discrete
        self.generator_nonlin_out_continuous = generator_nonlin_out_continuous
        self.generator_batch_norm = generator_batch_norm
        self.generator_dropout = generator_dropout
        self.generator_loss = generator_loss
        self.generator_lr = generator_lr
        self.generator_weight_decay = generator_weight_decay
        self.generator_residual = generator_residual
        self.discriminator_n_layers_hidden = discriminator_n_layers_hidden
        self.discriminator_n_units_hidden = discriminator_n_units_hidden
        self.discriminator_nonlin = discriminator_nonlin
        self.discriminator_n_iter = discriminator_n_iter
        self.discriminator_batch_norm = discriminator_batch_norm
        self.discriminator_dropout = discriminator_dropout
        self.discriminator_loss = discriminator_loss
        self.discriminator_lr = discriminator_lr
        self.discriminator_weight_decay = discriminator_weight_decay
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.random_state = random_state
        self.clipping_value = clipping_value
        self.mode = mode
        self.encoder_max_clusters = encoder_max_clusters
        self.encoder = encoder
        self.device = device
        self.gamma_penalty = gamma_penalty
        self.moments_penalty = moments_penalty
        self.embedding_penalty = embedding_penalty

    @staticmethod
    def name() -> str:
        return "ctimegan"

    @staticmethod
    def type() -> str:
        return "time_series"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return TimeGANPlugin.hyperparameter_space()

    def _fit(
        self, X: DataLoader, *args: Any, **kwargs: Any
    ) -> "ConditionalTimeGANPlugin":
        if X.type() not in ["time_series", "time_series_survival"]:
            raise ValueError("Invalid data type = {X.type()}")

        if X.type() == "time_series":
            static, temporal, temporal_horizons, outcome = X.unpack(pad=True)
        elif X.type() == "time_series_survival":
            static, temporal, temporal_horizons, T, E = X.unpack(pad=True)
            outcome = pd.concat([pd.Series(T), pd.Series(E)], axis=1)
            outcome.columns = ["time_to_event", "event"]

        self.conditional = TimeSeriesBinEncoder().fit_transform(
            pd.concat(
                [static.reset_index(drop=True), outcome.reset_index(drop=True)], axis=1
            ),
            temporal,
            temporal_horizons,
        )

        self.model = TimeGANPlugin(
            n_iter=self.n_iter,
            generator_n_layers_hidden=self.generator_n_layers_hidden,
            generator_n_units_hidden=self.generator_n_units_hidden,
            generator_nonlin=self.generator_nonlin,
            generator_nonlin_out_discrete=self.generator_nonlin_out_discrete,
            generator_nonlin_out_continuous=self.generator_nonlin_out_continuous,
            generator_batch_norm=self.generator_batch_norm,
            generator_dropout=self.generator_dropout,
            generator_loss=self.generator_loss,
            generator_lr=self.generator_lr,
            generator_weight_decay=self.generator_weight_decay,
            generator_residual=self.generator_residual,
            discriminator_n_layers_hidden=self.discriminator_n_layers_hidden,
            discriminator_n_units_hidden=self.discriminator_n_units_hidden,
            discriminator_nonlin=self.discriminator_nonlin,
            discriminator_n_iter=self.discriminator_n_iter,
            discriminator_batch_norm=self.discriminator_batch_norm,
            discriminator_dropout=self.discriminator_dropout,
            discriminator_loss=self.discriminator_loss,
            discriminator_lr=self.discriminator_lr,
            discriminator_weight_decay=self.discriminator_weight_decay,
            batch_size=self.batch_size,
            n_iter_print=self.n_iter_print,
            random_state=self.random_state,
            clipping_value=self.clipping_value,
            encoder_max_clusters=self.encoder_max_clusters,
            encoder=self.encoder,
            device=self.device,
            mode=self.mode,
            gamma_penalty=self.gamma_penalty,
            moments_penalty=self.moments_penalty,
            embedding_penalty=self.embedding_penalty,
            use_horizon_condition=True,
        )
        self.model.fit(X, cond=self.conditional)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        cond = self.conditional.sample(count, replace=True)

        # already calls safe generate
        return self.model._generate(count, syn_schema, cond=cond)


plugin = ConditionalTimeGANPlugin
