"""
Implementation for the paper "Time-series Generative Adversarial Networks", Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar
"""
# stdlib
from typing import Any, Callable, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.models.tabular_encoder import (
    TabularEncoder,
    TimeSeriesTabularEncoder,
)
from synthcity.plugins.core.models.ts_rnn import TimeSeriesRNN
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.generic import GenericPlugins
from synthcity.utils.constants import DEVICE


class ARModel:
    def __init__(
        self,
        n_iter: int = 1000,
        n_layers_hidden: int = 2,
        n_units_hidden: int = 150,
        nonlin: str = "leaky_relu",
        nonlin_out_discrete: str = "softmax",
        nonlin_out_continuous: str = "tanh",
        batch_norm: bool = False,
        dropout: float = 0.01,
        loss: Optional[Callable] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        residual: bool = True,
        batch_size: int = 64,
        n_iter_print: int = 10,
        random_state: int = 0,
        clipping_value: int = 0,
        encoder_max_clusters: int = 20,
        encoder: Any = None,
        device: Any = DEVICE,
        mode: str = "RNN",
        static_model: str = "ctgan",
    ) -> None:
        self.n_iter = n_iter
        self.n_layers_hidden = n_layers_hidden
        self.n_units_hidden = n_units_hidden
        self.nonlin = nonlin
        self.nonlin_out_discrete = nonlin_out_discrete
        self.nonlin_out_continuous = nonlin_out_continuous
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.residual = residual
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.random_state = random_state
        self.clipping_value = clipping_value
        self.mode = mode
        self.encoder_max_clusters = encoder_max_clusters
        self.encoder = encoder
        self.device = device

        self.static_model_name = static_model
        self.static_generator = GenericPlugins().get(
            self.static_model_name, device=self.device, n_iter=self.n_iter
        )

        self.temporal_encoder = TimeSeriesTabularEncoder(
            max_clusters=encoder_max_clusters
        )

    def fit(
        self,
        static: pd.DataFrame,
        temporal: List[pd.DataFrame],
        temporal_horizons: List,
    ) -> "ARModel":
        self.temporal_encoder.fit(static, temporal, temporal_horizons)
        (
            static_enc,
            temporal_enc,
            temporal_horizons_enc,
        ) = self.temporal_encoder.transform(static, temporal, temporal_horizons)

        self.temporal_encoded_columns = temporal_enc[0].columns
        self.static_encoded_columns = static_enc.columns
        self.temporal_columns = temporal[0].columns
        self.static_columns = static.columns
        self.temporal_len = len(temporal[0])

        temporal_enc = np.asarray(temporal_enc)
        static_enc = np.asarray(static_enc)

        outcome_enc = temporal_enc[:, -1, :].squeeze()
        temporal_enc = temporal_enc[:, :-1, :]

        # static generator
        temporal_init = temporal_enc[:, 0, :].squeeze()
        static_data_with_horizons = np.concatenate(
            [static_enc, np.asarray(temporal_horizons_enc), temporal_init], axis=1
        )

        self.static_generator = (
            GenericPlugins()
            .get(self.static_model_name, device=self.device, n_iter=self.n_iter)
            .fit(pd.DataFrame(static_data_with_horizons))
        )
        # temporal forecaster
        _, temporal_nonlin = self.temporal_encoder.activation_layout(
            discrete_activation="softmax",
            continuous_activation="tanh",
        )

        self.temporal_model = TimeSeriesRNN(
            task_type="regression",
            n_static_units_in=static_enc.shape[-1],
            n_temporal_units_in=temporal_enc[0].shape[-1],
            output_shape=outcome_enc.shape[1:],
            n_static_units_hidden=self.n_units_hidden,
            n_static_layers_hidden=self.n_layers_hidden,
            n_temporal_units_hidden=self.n_units_hidden,
            n_temporal_layers_hidden=self.n_layers_hidden,
            n_iter=self.n_iter,
            mode=self.mode,
            n_iter_print=self.n_iter_print,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            device=self.device,
            dropout=self.dropout,
            nonlin=self.nonlin,
            nonlin_out=temporal_nonlin,
        )
        self.temporal_model.fit(static_enc, temporal_enc, outcome_enc)
        return self

    def generate(self, count: int) -> Tuple:
        static_data_with_horizons = self.static_generator.generate(count).numpy()
        static_enc = static_data_with_horizons[:, : len(self.static_encoded_columns)]
        temporal_horizons_enc = static_data_with_horizons[
            :,
            len(self.static_encoded_columns) : len(self.static_encoded_columns)
            + self.temporal_len,
        ]
        temporal_init = static_data_with_horizons[
            :, len(self.static_encoded_columns) + self.temporal_len :
        ]

        temporal_enc = np.expand_dims(temporal_init, axis=1)

        # Temporal generation
        for horizon in range(1, self.temporal_len):
            next_temporal_enc = self.temporal_model.predict(
                np.asarray(static_enc), temporal_enc
            )
            next_temporal_enc = np.expand_dims(next_temporal_enc, axis=1)

            temporal_enc = np.concatenate([temporal_enc, next_temporal_enc], axis=1)

        assert temporal_enc.shape[1] == self.temporal_len
        # Decoding
        static_enc = pd.DataFrame(static_enc, columns=self.static_encoded_columns)
        temporal_enc_df = []
        for item in temporal_enc:
            temporal_enc_df.append(
                pd.DataFrame(item, columns=self.temporal_encoded_columns)
            )

        (
            static_raw,
            temporal_raw,
            temporal_horizons,
        ) = self.temporal_encoder.inverse_transform(
            static_enc, temporal_enc_df, temporal_horizons_enc.tolist()
        )

        static = pd.DataFrame(static_raw, columns=self.static_columns)
        temporal = []
        for item in temporal_raw:
            temporal.append(pd.DataFrame(item, columns=self.temporal_columns))

        return static, temporal, temporal_horizons


class AutoregressivePlugin(Plugin):
    """Synthetic time series generation using Autoregressive models.

    Args:
        n_iter: int
            Maximum number of iterations in the Generator.
        n_units_in: int
            Number of features
        n_layers_hidden: int
            Number of hidden layers in the generator
        n_units_hidden: int
            Number of hidden units in each layer of the Generator
        nonlin: string, default 'elu'
            Nonlinearity to use in the generator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        batch_norm: bool
            Enable/disable batch norm for the generator
        dropout: float
            Dropout value. If 0, the dropout is not used.
        residual: bool
            Use residuals for the generator
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
        static_model:
            Model used for static data generation

    Example:
        >>> from synthcity.plugins import Plugins
        >>> from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
        >>> from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
        >>>
        >>> plugin = Plugins().get("autoregressive")
        >>> static, temporal, outcome = GoogleStocksDataloader(as_numpy=True).load()
        >>> loader = TimeSeriesDataLoader(
        >>>             temporal_data=temporal_data,
        >>>             static_data=static_data,
        >>>             outcome=outcome,
        >>> )
        >>> plugin.fit(loader)
        >>> plugin.generate()
    """

    def __init__(
        self,
        n_iter: int = 1000,
        n_layers_hidden: int = 2,
        n_units_hidden: int = 150,
        nonlin: str = "leaky_relu",
        nonlin_out_discrete: str = "softmax",
        nonlin_out_continuous: str = "tanh",
        batch_norm: bool = False,
        dropout: float = 0.01,
        loss: Optional[Callable] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        residual: bool = True,
        batch_size: int = 64,
        n_iter_print: int = 10,
        random_state: int = 0,
        clipping_value: int = 0,
        encoder_max_clusters: int = 20,
        device: Any = DEVICE,
        mode: str = "RNN",
        static_model: str = "ctgan",
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.n_layers_hidden = n_layers_hidden
        self.n_units_hidden = n_units_hidden
        self.nonlin = nonlin
        self.nonlin_out_discrete = nonlin_out_discrete
        self.nonlin_out_continuous = nonlin_out_continuous
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.residual = residual
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.random_state = random_state
        self.clipping_value = clipping_value
        self.mode = mode
        self.encoder_max_clusters = encoder_max_clusters
        self.device = device

        self.ar_model = ARModel(
            n_units_hidden=self.n_units_hidden,
            n_layers_hidden=self.n_layers_hidden,
            n_iter=self.n_iter,
            mode=self.mode,
            n_iter_print=self.n_iter_print,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            device=self.device,
            dropout=self.dropout,
            nonlin=self.nonlin,
            encoder_max_clusters=encoder_max_clusters,
            static_model=static_model,
        )
        self.outcome_encoder = TabularEncoder(max_clusters=encoder_max_clusters)

    @staticmethod
    def name() -> str:
        return "autoregressive"

    @staticmethod
    def type() -> str:
        return "time_series"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="n_iter", low=100, high=1000, step=100),
            IntegerDistribution(name="n_layers_hidden", low=1, high=4),
            IntegerDistribution(name="n_units_hidden", low=50, high=150, step=50),
            CategoricalDistribution(
                name="nonlin", choices=["relu", "leaky_relu", "tanh", "elu"]
            ),
            FloatDistribution(name="dropout", low=0, high=0.2),
            CategoricalDistribution(name="lr", choices=[1e-3, 2e-4, 1e-4]),
            CategoricalDistribution(name="weight_decay", choices=[1e-3, 1e-4]),
            CategoricalDistribution(name="batch_size", choices=[100, 200, 500]),
            IntegerDistribution(name="encoder_max_clusters", low=2, high=20),
            CategoricalDistribution(name="mode", choices=["LSTM", "GRU", "RNN"]),
            FloatDistribution(name="gamma_penalty", low=0, high=1000),
            FloatDistribution(name="moments_penalty", low=0, high=1000),
            FloatDistribution(name="embedding_penalty", low=0, high=1000),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "AutoregressivePlugin":
        assert X.type() in ["time_series", "time_series_survival"]

        self.data_type = X.type()

        if self.data_type == "time_series":
            static, temporal, temporal_horizons, outcome = X.unpack()
        else:
            static, temporal, temporal_horizons, T, E = X.unpack()
            outcome = pd.concat([pd.Series(T), pd.Series(E)], axis=1)
            outcome.columns = ["time_to_event", "event"]

        # Train the static and temporal generator
        self.ar_model.fit(static, temporal, temporal_horizons)

        # Outcome generation
        self.outcome_encoder.fit(outcome)
        outcome_enc = self.outcome_encoder.transform(outcome)

        self.outcome_model = TimeSeriesRNN(
            task_type="regression",
            n_static_units_in=static.shape[-1],
            n_temporal_units_in=temporal[0].shape[-1],
            output_shape=outcome_enc.shape[1:],
            n_static_units_hidden=self.n_units_hidden,
            n_static_layers_hidden=self.n_layers_hidden,
            n_temporal_units_hidden=self.n_units_hidden,
            n_temporal_layers_hidden=self.n_layers_hidden,
            n_iter=self.n_iter,
            mode=self.mode,
            n_iter_print=self.n_iter_print,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            device=self.device,
            dropout=self.dropout,
            nonlin=self.nonlin,
            nonlin_out=self.outcome_encoder.activation_layout(
                discrete_activation="softmax",
                continuous_activation="tanh",
            ),
        )
        self.outcome_model.fit(
            np.asarray(static), np.asarray(temporal), np.asarray(outcome_enc)
        )
        self.outcome_encoded_columns = outcome_enc.columns

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _sample(count: int) -> Tuple:
            # Static and Temporal generation
            static, temporal, temporal_horizons = self.ar_model.generate(count)

            # Outcome generation
            outcome_enc = pd.DataFrame(
                self.outcome_model.predict(np.asarray(static), np.asarray(temporal)),
                columns=self.outcome_encoded_columns,
            )

            # Decoding
            outcome_raw = self.outcome_encoder.inverse_transform(outcome_enc)
            outcome = pd.DataFrame(
                outcome_raw, columns=self.data_info["outcome_features"]
            )

            return static, temporal, temporal_horizons, outcome

        return self._safe_generate_time_series(_sample, count, syn_schema)


plugin = AutoregressivePlugin
