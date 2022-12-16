"""
Implementation for the paper "Time-series Generative Adversarial Networks", Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar
"""
# stdlib
from typing import Any, Callable, List, Optional, Tuple, Union

# third party
import numpy as np
import pandas as pd

# synthcity absolute
import synthcity.logger as log
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.models.tabular_encoder import BinEncoder, TabularEncoder
from synthcity.plugins.core.models.ts_model import TimeSeriesModel, modes
from synthcity.plugins.core.models.ts_tabular_gan import TimeSeriesTabularGAN
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE
from synthcity.utils.samplers import ImbalancedDatasetSampler


class TimeGANPlugin(Plugin):
    """Synthetic time series generation using TimeGAN.

    Args:
        n_iter: int
            Maximum number of iterations in the Generator.
        n_units_conditional: int = 0,
            Number of conditional units
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
        >>> plugin = Plugins().get("timegan", n_iter = 50)
        >>> plugin.fit(loader)
        >>>
        >>> plugin.generate(count = 10)

    Reference: "Time-series Generative Adversarial Networks", Jinsung Yoon et. al
    """

    def __init__(
        self,
        n_iter: int = 1000,
        n_units_conditional: int = 0,
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
        use_horizon_condition: bool = True,
        dataloader_sampling_strategy: str = "imbalanced_time_censoring",  # none, imbalanced_censoring, imbalanced_time_censoring
        **kwargs: Any,
    ) -> None:
        super().__init__()

        log.info(
            f"""TimeGAN: mode = {mode} dataloader_sampling_strategy = {dataloader_sampling_strategy}"""
        )
        self.n_iter = n_iter
        self.n_units_conditional = n_units_conditional
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
        self.use_horizon_condition = use_horizon_condition
        self.dataloader_sampling_strategy = dataloader_sampling_strategy

        self.outcome_encoder = TabularEncoder(max_clusters=encoder_max_clusters)

    @staticmethod
    def name() -> str:
        return "timegan"

    @staticmethod
    def type() -> str:
        return "time_series"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="n_iter", low=100, high=1000, step=100),
            IntegerDistribution(name="generator_n_layers_hidden", low=1, high=4),
            IntegerDistribution(
                name="generator_n_units_hidden", low=50, high=150, step=50
            ),
            CategoricalDistribution(
                name="generator_nonlin", choices=["relu", "leaky_relu", "tanh", "elu"]
            ),
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
            CategoricalDistribution(name="mode", choices=modes),
            FloatDistribution(name="gamma_penalty", low=0, high=1000),
            FloatDistribution(name="moments_penalty", low=0, high=1000),
            FloatDistribution(name="embedding_penalty", low=0, high=1000),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "TimeGANPlugin":
        if X.type() not in ["time_series", "time_series_survival"]:
            raise ValueError(f"Invalid data type = {X.type()}")

        cond: Optional[Union[pd.DataFrame, pd.Series]] = None
        sampler: Optional[ImbalancedDatasetSampler] = None

        if self.n_units_conditional > 0:
            if "cond" not in kwargs:
                raise ValueError("expecting 'cond' for training")
            cond = kwargs["cond"]

        # Static and temporal generation
        if X.type() == "time_series":
            static, temporal, temporal_horizons, outcome = X.unpack(pad=True)
        elif X.type() == "time_series_survival":
            static, temporal, temporal_horizons, T, E = X.unpack(pad=True)
            outcome = pd.concat([pd.Series(T), pd.Series(E)], axis=1)
            outcome.columns = ["time_to_event", "event"]

            sampling_labels: Optional[list] = None

            if self.dataloader_sampling_strategy == "imbalanced_censoring":
                sampling_labels = list(E.values)
            elif self.dataloader_sampling_strategy == "imbalanced_time_censoring":
                Tbins = (
                    BinEncoder().fit_transform(T.to_frame()).values.squeeze().tolist()
                )
                sampling_labels = list(zip(E, Tbins))

            if sampling_labels is not None:
                sampler = ImbalancedDatasetSampler(sampling_labels)

        self.cov_model = TimeSeriesTabularGAN(
            static_data=static,
            temporal_data=temporal,
            temporal_horizons=temporal_horizons,
            generator_n_iter=self.n_iter,
            n_units_conditional=self.n_units_conditional,
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
            mode=self.mode,
            encoder_max_clusters=self.encoder_max_clusters,
            encoder=self.encoder,
            device=self.device,
            gamma_penalty=self.gamma_penalty,
            moments_penalty=self.moments_penalty,
            embedding_penalty=self.embedding_penalty,
            use_horizon_condition=self.use_horizon_condition,
            dataloader_sampler=sampler,
        )
        self.cov_model.fit(static, temporal, temporal_horizons, cond=cond)

        # Outcome generation
        self.outcome_encoder.fit(outcome)
        outcome_enc = self.outcome_encoder.transform(outcome)
        self.outcome_encoded_columns = outcome_enc.columns

        self.outcome_model = TimeSeriesModel(
            task_type="regression",
            n_static_units_in=static.shape[-1],
            n_temporal_units_in=temporal[0].shape[-1],
            n_temporal_window=temporal[0].shape[0],
            output_shape=outcome_enc.shape[1:],
            n_static_units_hidden=self.generator_n_units_hidden,
            n_static_layers_hidden=self.generator_n_layers_hidden,
            n_temporal_units_hidden=self.generator_n_units_hidden,
            n_temporal_layers_hidden=self.generator_n_layers_hidden,
            n_iter=self.n_iter,
            mode=self.mode,
            n_iter_print=self.n_iter_print,
            batch_size=self.batch_size,
            lr=self.generator_lr,
            weight_decay=self.generator_weight_decay,
            window_size=1,
            device=self.device,
            dropout=self.generator_dropout,
            nonlin=self.generator_nonlin,
            nonlin_out=self.outcome_encoder.activation_layout(
                discrete_activation="softmax",
                continuous_activation="tanh",
            ),
        )
        self.outcome_model.fit(
            np.asarray(static),
            np.asarray(temporal),
            np.asarray(temporal_horizons),
            np.asarray(outcome_enc),
        )

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        cond: Optional[Union[pd.DataFrame, pd.Series]] = None
        static_data_cond: Optional[pd.DataFrame] = None
        temporal_horizons_cond: Optional[list] = None

        if "cond" in kwargs:
            cond = kwargs["cond"]
        if "static_data" in kwargs:
            static_data_cond = kwargs["static_data"]
        if "temporal_horizons" in kwargs:
            temporal_horizons_cond = kwargs["temporal_horizons"]

        def _sample(count: int) -> Tuple:
            local_cond: Optional[Union[pd.DataFrame, pd.Series]] = None
            local_static_data: Optional[pd.DataFrame] = None
            local_temporal_horizons: Optional[list] = None
            if cond is not None:
                local_cond = cond.sample(count, replace=True)
            if static_data_cond is not None:
                local_static_data = static_data_cond.sample(count, replace=True)
            if temporal_horizons_cond is not None:
                ids = list(range(len(temporal_horizons_cond)))
                local_ids = np.random.choice(ids, count, replace=True)
                local_temporal_horizons = np.asarray(temporal_horizons_cond)[
                    local_ids
                ].tolist()

            static, temporal, temporal_horizons = self.cov_model.generate(
                count,
                cond=local_cond,
                static_data=local_static_data,
                temporal_horizons=local_temporal_horizons,
            )

            outcome_enc = pd.DataFrame(
                self.outcome_model.predict(
                    np.asarray(static),
                    np.asarray(temporal),
                    np.asarray(temporal_horizons),
                ),
                columns=self.outcome_encoded_columns,
            )
            outcome_raw = self.outcome_encoder.inverse_transform(outcome_enc)
            outcome = pd.DataFrame(
                outcome_raw, columns=self.data_info["outcome_features"]
            )

            if self.data_info["data_type"] == "time_series":
                return static, temporal, temporal_horizons, outcome
            elif self.data_info["data_type"] == "time_series_survival":
                return (
                    static,
                    temporal,
                    temporal_horizons,
                    outcome[self.data_info["time_to_event_column"]],
                    outcome[self.data_info["event_column"]],
                )
            else:
                raise RuntimeError("unknow data type")

        return self._safe_generate_time_series(_sample, count, syn_schema)


plugin = TimeGANPlugin
