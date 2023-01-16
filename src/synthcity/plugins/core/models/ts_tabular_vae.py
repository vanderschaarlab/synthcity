# stdlib
from typing import Any, List, Tuple

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments

# synthcity absolute
from synthcity.utils.constants import DEVICE

# synthcity relative
from .tabular_encoder import TimeSeriesTabularEncoder
from .ts_vae import TimeSeriesVAE


class TimeSeriesTabularVAE(torch.nn.Module):
    """
    .. inheritance-diagram:: synthcity.plugins.core.models.ts_tabular_vae.TimeSeriesTabularVAE
        :parts: 1

    TimeSeries Tabular Variational AutoEncoder implementation.

    This class combines TimeSeriesVAE and tabular encoder to form a generative model for tabular data.

    Args:
        static_data: pd.DataFrame
            Reference static data
        temporal_data: List[pd.DataFrame]
            Reference temporal data
        observation_times: List
            Reference temporal horizons
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
        encoder_max_clusters: int
            The max number of clusters to create for continuous columns when encoding
        encoder:
            Pre-trained tabular encoder. If None, a new encoder is trained.
        device:
            Device to use for computation
        loss_factor: float
            Reconstruction loss weight.
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
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        static_data: pd.DataFrame,
        temporal_data: List[pd.DataFrame],
        observation_times: List,
        decoder_n_layers_hidden: int = 2,
        decoder_n_units_hidden: int = 150,
        decoder_nonlin: str = "leaky_relu",
        decoder_nonlin_out_discrete: str = "softmax",
        decoder_nonlin_out_continuous: str = "tanh",
        decoder_batch_norm: bool = False,
        decoder_dropout: float = 0.01,
        decoder_weight_decay: float = 1e-3,
        decoder_residual: bool = True,
        encoder_n_layers_hidden: int = 3,
        encoder_n_units_hidden: int = 300,
        encoder_nonlin: str = "leaky_relu",
        encoder_batch_norm: bool = False,
        encoder_dropout: float = 0.1,
        weight_decay: float = 1e-3,
        n_iter: int = 1,
        lr: float = 1e-3,
        batch_size: int = 64,
        n_iter_print: int = 50,
        random_state: int = 0,
        clipping_value: int = 0,
        mode: str = "RNN",
        encoder_max_clusters: int = 20,
        encoder: Any = None,
        device: Any = DEVICE,
        loss_factor: int = 2,
    ) -> None:
        super(TimeSeriesTabularVAE, self).__init__()
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = TimeSeriesTabularEncoder(
                max_clusters=encoder_max_clusters
            ).fit(static_data, temporal_data, observation_times)

        self.static_columns = static_data.columns
        self.temporal_columns = temporal_data[0].columns

        n_static_units, n_temporal_units = self.encoder.n_features()
        static_act, temporal_act = self.encoder.activation_layout(
            discrete_activation=decoder_nonlin_out_discrete,
            continuous_activation=decoder_nonlin_out_continuous,
        )
        self.model = TimeSeriesVAE(
            n_static_units=n_static_units,
            n_static_units_embedding=n_static_units,
            n_temporal_units=n_temporal_units,
            n_temporal_window=len(temporal_data[0]),
            n_temporal_units_embedding=n_temporal_units,
            batch_size=batch_size,
            decoder_n_layers_hidden=decoder_n_layers_hidden,
            decoder_n_units_hidden=decoder_n_units_hidden,
            decoder_nonlin=decoder_nonlin,
            decoder_static_nonlin_out=static_act,
            decoder_temporal_nonlin_out=temporal_act,
            decoder_batch_norm=decoder_batch_norm,
            decoder_dropout=decoder_dropout,
            decoder_residual=decoder_residual,
            decoder_mode=mode,
            encoder_n_units_hidden=encoder_n_units_hidden,
            encoder_n_layers_hidden=encoder_n_layers_hidden,
            encoder_nonlin=encoder_nonlin,
            encoder_batch_norm=encoder_batch_norm,
            encoder_dropout=encoder_dropout,
            encoder_mode=mode,
            lr=lr,
            weight_decay=weight_decay,
            n_iter=n_iter,
            clipping_value=clipping_value,
            n_iter_print=n_iter_print,
            random_state=random_state,
            device=device,
            loss_factor=loss_factor,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def encode(
        self,
        static_data: pd.DataFrame,
        temporal_data: List[pd.DataFrame],
        observation_times: List,
    ) -> Tuple:
        return self.encoder.transform(static_data, temporal_data, observation_times)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def decode(
        self,
        static_data: pd.DataFrame,
        temporal_data: List[pd.DataFrame],
        observation_times: List,
    ) -> Tuple:
        return self.encoder.inverse_transform(
            static_data, temporal_data, observation_times
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def encode_static(
        self,
        static_data: pd.DataFrame,
    ) -> Tuple:
        return self.encoder.transform_static(static_data)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def decode_static(
        self,
        static_data: pd.DataFrame,
    ) -> Tuple:
        return self.encoder.inverse_transform_static(static_data)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def encode_horizons(
        self,
        observation_times: List,
    ) -> Tuple:
        return self.encoder.transform_observation_times(observation_times)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def decode_horizons(
        self,
        observation_times: List,
    ) -> Tuple:
        return self.encoder.inverse_transform_observation_times(observation_times)

    def get_encoder(self) -> TimeSeriesTabularEncoder:
        return self.encoder

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        static_data: pd.DataFrame,
        temporal_data: List[pd.DataFrame],
        observation_times: List,
        encoded: bool = False,
    ) -> Any:
        if encoded:
            static_enc = static_data
            temporal_enc = temporal_data
            observation_times_enc = observation_times
        else:
            static_enc, temporal_enc, observation_times_enc = self.encode(
                static_data, temporal_data, observation_times
            )

        self.static_encoded_columns = static_enc.columns
        self.temporal_encoded_columns = temporal_enc[0].columns

        self.model.fit(
            np.asarray(static_enc),
            np.asarray(temporal_enc),
            np.asarray(observation_times_enc),
        )
        return self

    def generate(
        self,
        count: int,
    ) -> pd.DataFrame:
        static_raw, temporal_raw, observation_times = self.model.generate(
            count,
        )

        static_data = pd.DataFrame(static_raw, columns=self.static_encoded_columns)
        temporal_data = []
        for item in temporal_raw:
            temporal_data.append(
                pd.DataFrame(item, columns=self.temporal_encoded_columns)
            )

        return self.decode(static_data, temporal_data, observation_times.tolist())

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, count: int) -> torch.Tensor:
        return self.model.forward(count)
