# stdlib
from pathlib import Path
from typing import Any, List, Tuple

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
from synthcity.plugins.core.models.tabular_encoder import TabularEncoder
from synthcity.plugins.core.models.ts_model import TimeSeriesModel, modes
from synthcity.plugins.core.models.ts_tabular_vae import TimeSeriesTabularVAE
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE


class TimeVAEPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.time_series.plugin_timevae.TimeVAEPlugin
        :parts: 1


    Synthetic time series generation using a Variational AutoEncoder.

    Args:
        n_iter: int
            Maximum number of iterations in the decoder.
        n_units_in: int
            Number of features
        decoder_n_layers_hidden: int
            Number of hidden layers in the decoder
        decoder_n_units_hidden: int
            Number of hidden units in each layer of the decoder
        decoder_nonlin: string, default 'elu'
            Nonlinearity to use in the decoder. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        decoder_batch_norm: bool
            Enable/disable batch norm for the decoder
        decoder_dropout: float
            Dropout value. If 0, the dropout is not used.
        decoder_residual: bool
            Use residuals for the decoder
        encoder_n_layers_hidden: int
            Number of hidden layers in the encoder
        encoder_n_units_hidden: int
            Number of hidden units in each layer of the encoder
        encoder_nonlin: string, default 'relu'
            Nonlinearity to use in the encoder. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        encoder_n_iter: int
            Maximum number of iterations in the encoder.
        encoder_batch_norm: bool
            Enable/disable batch norm for the encoder
        encoder_dropout: float
            Dropout value for the encoder. If 0, the dropout is not used.
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
        mode: str = "RNN"
            Core neural net architecture.
            Available models:
                - "LSTM"
                - "GRU"
                - "RNN"
                - "Transformer"
                - "MLSTM_FCN"
                - "TCN"
                - "InceptionTime"
                - "InceptionTimePlus"
                - "XceptionTime"
                - "ResCNN"
                - "OmniScaleCNN"
                - "XCM"
        device
            The device used by PyTorch. cpu/cuda
        use_horizon_condition: bool. Default = True
            Whether to condition the covariate generation on the observation times or not.
        encoder_max_clusters: int
            The max number of clusters to create for continuous columns when encoding
        encoder:
            Pre-trained tabular encoder. If None, a new encoder is trained.
        # Core Plugin arguments
        workspace: Path.
            Optional Path for caching intermediary results.
        compress_dataset: bool. Default = False.
            Drop redundant features before training the generator.
        sampling_patience: int.
            Max inference iterations to wait for the generated data to match the training schema.


    Example:
        >>> from synthcity.plugins import Plugins
        >>> from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
        >>> from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
        >>>
        >>> plugin = Plugins().get("timevae")
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
        decoder_n_layers_hidden: int = 2,
        decoder_n_units_hidden: int = 150,
        decoder_nonlin: str = "leaky_relu",
        decoder_nonlin_out_discrete: str = "softmax",
        decoder_nonlin_out_continuous: str = "tanh",
        decoder_batch_norm: bool = False,
        decoder_dropout: float = 0.01,
        decoder_residual: bool = True,
        encoder_n_layers_hidden: int = 3,
        encoder_n_units_hidden: int = 300,
        encoder_nonlin: str = "leaky_relu",
        encoder_batch_norm: bool = False,
        encoder_dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 64,
        n_iter_print: int = 10,
        clipping_value: int = 0,
        encoder_max_clusters: int = 20,
        encoder: Any = None,
        device: Any = DEVICE,
        mode: str = "LSTM",
        gamma_penalty: float = 1,
        moments_penalty: float = 100,
        embedding_penalty: float = 10,
        # core plugin arguments
        random_state: int = 0,
        workspace: Path = Path("workspace"),
        compress_dataset: bool = False,
        sampling_patience: int = 500,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            device=device,
            random_state=random_state,
            sampling_patience=sampling_patience,
            workspace=workspace,
            compress_dataset=compress_dataset,
        )

        self.n_iter = n_iter
        self.lr = lr
        self.weight_decay = weight_decay
        self.decoder_n_layers_hidden = decoder_n_layers_hidden
        self.decoder_n_units_hidden = decoder_n_units_hidden
        self.decoder_nonlin = decoder_nonlin
        self.decoder_nonlin_out_discrete = decoder_nonlin_out_discrete
        self.decoder_nonlin_out_continuous = decoder_nonlin_out_continuous
        self.decoder_batch_norm = decoder_batch_norm
        self.decoder_dropout = decoder_dropout
        self.decoder_residual = decoder_residual
        self.encoder_n_layers_hidden = encoder_n_layers_hidden
        self.encoder_n_units_hidden = encoder_n_units_hidden
        self.encoder_nonlin = encoder_nonlin
        self.encoder_batch_norm = encoder_batch_norm
        self.encoder_dropout = encoder_dropout
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

        self.outcome_encoder = TabularEncoder(max_clusters=encoder_max_clusters)

    @staticmethod
    def name() -> str:
        return "timevae"

    @staticmethod
    def type() -> str:
        return "time_series"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="n_iter", low=100, high=1000, step=100),
            IntegerDistribution(name="decoder_n_layers_hidden", low=1, high=4),
            IntegerDistribution(
                name="decoder_n_units_hidden", low=50, high=150, step=50
            ),
            CategoricalDistribution(
                name="decoder_nonlin", choices=["relu", "leaky_relu", "tanh", "elu"]
            ),
            FloatDistribution(name="decoder_dropout", low=0, high=0.2),
            IntegerDistribution(name="encoder_n_layers_hidden", low=1, high=4),
            IntegerDistribution(
                name="encoder_n_units_hidden", low=50, high=150, step=50
            ),
            CategoricalDistribution(
                name="encoder_nonlin",
                choices=["relu", "leaky_relu", "tanh", "elu"],
            ),
            IntegerDistribution(name="encoder_n_iter", low=1, high=5),
            FloatDistribution(name="encoder_dropout", low=0, high=0.2),
            CategoricalDistribution(name="lr", choices=[1e-3, 2e-4, 1e-4]),
            CategoricalDistribution(name="weight_decay", choices=[1e-3, 1e-4]),
            CategoricalDistribution(name="batch_size", choices=[100, 200, 500]),
            IntegerDistribution(name="encoder_max_clusters", low=2, high=20),
            CategoricalDistribution(name="mode", choices=modes),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "TimeVAEPlugin":
        if X.type() not in ["time_series", "time_series_survival"]:
            raise ValueError(f"Invalid data type = {X.type()}")

        # Static and temporal generation
        if X.type() == "time_series":
            static, temporal, observation_times, outcome = X.unpack(pad=True)
        elif X.type() == "time_series_survival":
            static, temporal, observation_times, T, E = X.unpack(pad=True)
            outcome = pd.concat([pd.Series(T), pd.Series(E)], axis=1)
            outcome.columns = ["time_to_event", "event"]

        self.cov_model = TimeSeriesTabularVAE(
            static_data=static,
            temporal_data=temporal,
            observation_times=observation_times,
            n_iter=self.n_iter,
            lr=self.lr,
            weight_decay=self.weight_decay,
            decoder_n_layers_hidden=self.decoder_n_layers_hidden,
            decoder_n_units_hidden=self.decoder_n_units_hidden,
            decoder_nonlin=self.decoder_nonlin,
            decoder_nonlin_out_discrete=self.decoder_nonlin_out_discrete,
            decoder_nonlin_out_continuous=self.decoder_nonlin_out_continuous,
            decoder_batch_norm=self.decoder_batch_norm,
            decoder_dropout=self.decoder_dropout,
            decoder_residual=self.decoder_residual,
            encoder_n_layers_hidden=self.encoder_n_layers_hidden,
            encoder_n_units_hidden=self.encoder_n_units_hidden,
            encoder_nonlin=self.encoder_nonlin,
            encoder_batch_norm=self.encoder_batch_norm,
            encoder_dropout=self.encoder_dropout,
            batch_size=self.batch_size,
            n_iter_print=self.n_iter_print,
            random_state=self.random_state,
            clipping_value=self.clipping_value,
            mode=self.mode,
            encoder_max_clusters=self.encoder_max_clusters,
            encoder=self.encoder,
            device=self.device,
        )
        self.cov_model.fit(static, temporal, observation_times)

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
            n_static_units_hidden=self.decoder_n_units_hidden,
            n_static_layers_hidden=self.decoder_n_layers_hidden,
            n_temporal_units_hidden=self.decoder_n_units_hidden,
            n_temporal_layers_hidden=self.decoder_n_layers_hidden,
            n_iter=self.n_iter,
            mode=self.mode,
            n_iter_print=self.n_iter_print,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            window_size=1,
            device=self.device,
            dropout=self.decoder_dropout,
            nonlin=self.decoder_nonlin,
            nonlin_out=self.outcome_encoder.activation_layout(
                discrete_activation="softmax",
                continuous_activation="tanh",
            ),
        )
        self.outcome_model.fit(
            np.asarray(static),
            np.asarray(temporal),
            np.asarray(observation_times),
            np.asarray(outcome_enc),
        )

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _sample(count: int) -> Tuple:
            static, temporal, observation_times = self.cov_model.generate(count)

            outcome_enc = pd.DataFrame(
                self.outcome_model.predict(
                    np.asarray(static),
                    np.asarray(temporal),
                    np.asarray(observation_times),
                ),
                columns=self.outcome_encoded_columns,
            )
            outcome_raw = self.outcome_encoder.inverse_transform(outcome_enc)
            outcome = pd.DataFrame(
                outcome_raw, columns=self.data_info["outcome_features"]
            )

            if self.data_info["data_type"] == "time_series":
                return static, temporal, observation_times, outcome
            elif self.data_info["data_type"] == "time_series_survival":
                return (
                    static,
                    temporal,
                    observation_times,
                    outcome[self.data_info["time_to_event_column"]],
                    outcome[self.data_info["event_column"]],
                )
            else:
                raise RuntimeError("unknow data type")

        return self._safe_generate_time_series(_sample, count, syn_schema)


plugin = TimeVAEPlugin
