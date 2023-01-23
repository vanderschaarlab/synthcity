# stdlib
from typing import Any, Callable, List, Optional, Tuple, Union

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments

# synthcity absolute
from synthcity.utils.constants import DEVICE

# synthcity relative
from .tabular_encoder import TimeSeriesTabularEncoder
from .ts_gan import TimeSeriesGAN


class TimeSeriesTabularGAN(torch.nn.Module):
    """

    .. inheritance-diagram:: synthcity.plugins.core.models.ts_tabular_gan.TimeSeriesTabularGAN
        :parts: 1


    TimeSeries Tabular GAN implementation.

    This class combines TimeSeriesGAN and tabular encoder to form a generative model for tabular data.

    Args:
        static_data: pd.DataFrame,
            Reference static data
        temporal_data: List[pd.DataFrame],
            Reference temporal data
        observation_times: List
            Reference temporal horizons
        cond: Optional
            Optional conditional
        generator_n_layers_hidden: int. Default: 1
            Number of hidden layers in the generator
        generator_n_units_hidden: int. Default: 250
            Number of hidden units in each layer of the Generator
        generator_nonlin: string, default 'elu'
            Nonlinearity to use in the generator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        generator_n_iter: int. Default: 1000
            Maximum number of iterations in the Generator.
        generator_batch_norm: bool. Default: False
            Enable/disable batch norm for the generator
        generator_dropout: float. Default: 0
            Generator Dropout value. If 0, the dropout is not used.
        generator_residual: bool. Default: True
            Use residuals for the generator
        generator_lr: float. Default: 2e-4
            Generator learning rate.
        generator_weight_decay: float. Default = 1e-3
            l2 (ridge) penalty for the generator weights.
        discriminator_n_layers_hidden: int. Default = 1
            Number of hidden layers in the discriminator
        discriminator_n_units_hidden: int. Default = 300
            Number of hidden units in each layer of the discriminator
        discriminator_nonlin: string, default = 'relu'
            Nonlinearity to use in the discriminator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        discriminator_n_iter: int
            Maximum number of iterations in the discriminator.
        discriminator_batch_norm: bool. Default: False
            Enable/disable batch norm for the discriminator
        discriminator_dropout: float. Default = 0.1
            Dropout value for the discriminator. If 0, the dropout is not used.
        discriminator_lr: float. Default = 2e-4
            learning rate for discriminator optimizer. step_size equivalent in the JAX version.
        discriminator_weight_decay: float. Default = 1e-3
            l2 (ridge) penalty for the discriminator weights.
        batch_size: int. Default = 64
            Batch size
        n_iter_print: int
            Number of iterations after which to print updates and check the validation loss.
        random_state: int
            random_state used
        clipping_value: int, default = 0
            Gradients clipping value. Zero disables the feature
        gamma_penalty: float. Default = 1.
            Latent representation penalty
        moments_penalty: float. Default = 100.
            Generator Moments(var and mean) penalty
        embedding_penalty: float. Default = 10
            Embedding representation penalty
        dataloader_sampler: Optional[sampler.Sampler] = None
            Optional data sampler
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
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        static_data: pd.DataFrame,
        temporal_data: List[pd.DataFrame],
        observation_times: List,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 150,
        generator_nonlin: str = "leaky_relu",
        generator_nonlin_out_discrete: str = "softmax",
        generator_nonlin_out_continuous: str = "tanh",
        generator_n_iter: int = 1000,
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
        n_iter_print: int = 50,
        random_state: int = 0,
        clipping_value: int = 0,
        mode: str = "RNN",
        encoder_max_clusters: int = 20,
        encoder: Any = None,
        dataloader_sampler: Optional[torch.utils.data.sampler.Sampler] = None,
        device: Any = DEVICE,
        gamma_penalty: float = 1,
        moments_penalty: float = 100,
        embedding_penalty: float = 10,
        use_horizon_condition: bool = True,
    ) -> None:
        super(TimeSeriesTabularGAN, self).__init__()
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = TimeSeriesTabularEncoder(
                max_clusters=encoder_max_clusters
            ).fit(static_data, temporal_data, observation_times)

        self.static_columns = static_data.columns
        self.temporal_columns = temporal_data[0].columns
        n_units_conditional = 0 if cond is None else cond.shape[-1]

        n_static_units, n_temporal_units = self.encoder.n_features()
        static_act, temporal_act = self.encoder.activation_layout(
            discrete_activation=generator_nonlin_out_discrete,
            continuous_activation=generator_nonlin_out_continuous,
        )
        self.model = TimeSeriesGAN(
            n_static_units=n_static_units,
            n_static_units_latent=n_static_units,
            n_temporal_units=n_temporal_units,
            n_temporal_window=len(temporal_data[0]),
            n_temporal_units_latent=n_temporal_units,
            n_units_conditional=n_units_conditional,
            batch_size=batch_size,
            generator_n_layers_hidden=generator_n_layers_hidden,
            generator_n_units_hidden=generator_n_units_hidden,
            generator_nonlin=generator_nonlin,
            generator_static_nonlin_out=static_act,
            generator_temporal_nonlin_out=temporal_act,
            generator_n_iter=generator_n_iter,
            generator_batch_norm=generator_batch_norm,
            generator_dropout=generator_dropout,
            generator_loss=generator_loss,
            generator_lr=generator_lr,
            generator_residual=generator_residual,
            generator_weight_decay=generator_weight_decay,
            discriminator_n_units_hidden=discriminator_n_units_hidden,
            discriminator_n_layers_hidden=discriminator_n_layers_hidden,
            discriminator_n_iter=discriminator_n_iter,
            discriminator_nonlin=discriminator_nonlin,
            discriminator_batch_norm=discriminator_batch_norm,
            discriminator_dropout=discriminator_dropout,
            discriminator_loss=discriminator_loss,
            discriminator_lr=discriminator_lr,
            discriminator_weight_decay=discriminator_weight_decay,
            mode=mode,
            clipping_value=clipping_value,
            n_iter_print=n_iter_print,
            random_state=random_state,
            dataloader_sampler=dataloader_sampler,
            device=device,
            gamma_penalty=gamma_penalty,
            moments_penalty=moments_penalty,
            embedding_penalty=embedding_penalty,
            use_horizon_condition=use_horizon_condition,
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
        cond: Optional[Union[pd.DataFrame, pd.Series]] = None,
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
            np.asarray(cond),
        )
        return self

    def generate(
        self,
        count: int,
        cond: Optional[pd.DataFrame] = None,
        static_data: Optional[np.ndarray] = None,
        observation_times: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        if static_data is not None:
            static_data = self.encode_static(static_data)
        if observation_times is not None:
            observation_times = self.encode_horizons(observation_times)

        static_raw, temporal_raw, observation_times = self.model.generate(
            count,
            cond=cond,
            static_data=static_data,
            observation_times=observation_times,
        )

        static_data = pd.DataFrame(static_raw, columns=self.static_encoded_columns)
        temporal_data = []
        for item in temporal_raw:
            temporal_data.append(
                pd.DataFrame(item, columns=self.temporal_encoded_columns)
            )

        return self.decode(static_data, temporal_data, observation_times.tolist())

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, count: int, cond: Optional[pd.DataFrame] = None) -> torch.Tensor:
        return self.model.forward(count, cond)
