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
    TimeSeries Tabular GAN implementation.

    Args:
        n_static_units: int,
            Number of units for the static features
        n_static_units_latent: int,
            Number of latent units for the static features
        n_temporal_units: int,
            Number of units for the temporal features
        n_temporal_window: int,
            Number of temporal sequences for each subject
        n_temporal_units_latent: int,
            Number of temporal latent units
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
        generator_n_iter: int
            Maximum number of iterations in the Generator.
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
        seed: int
            Seed used
        n_iter_min: int
            Minimum number of iterations to go through before starting early stopping
        clipping_value: int, default 0
            Gradients clipping value
        lambda_gradient_penalty: float
            Lambda weight for the gradient penalty
        lambda_identifiability_penalty: float
            Lambda weight for the identifiability loss
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
        n_static_units_latent: int,
        n_temporal_window: int,
        n_temporal_units_latent: int,
        n_units_conditional: int = 0,
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
        generator_opt_betas: tuple = (0.9, 0.999),
        generator_residual: bool = True,
        generator_extra_penalties: list = [],  # "gradient_penalty", "identifiability_penalty"
        discriminator_n_layers_hidden: int = 3,
        discriminator_n_units_hidden: int = 300,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_batch_norm: bool = False,
        discriminator_dropout: float = 0.1,
        discriminator_loss: Optional[Callable] = None,
        discriminator_lr: float = 1e-3,
        discriminator_weight_decay: float = 1e-3,
        discriminator_opt_betas: tuple = (0.9, 0.999),
        discriminator_extra_penalties: list = [
            "gradient_penalty"
        ],  # "identifiability_penalty", "gradient_penalty"
        batch_size: int = 64,
        n_iter_print: int = 50,
        seed: int = 0,
        n_iter_min: int = 100,
        clipping_value: int = 0,
        lambda_gradient_penalty: float = 10,
        lambda_identifiability_penalty: float = 0.1,
        mode: str = "RNN",
        encoder_max_clusters: int = 20,
        encoder: Any = None,
        dataloader_sampler: Optional[torch.utils.data.sampler.Sampler] = None,
        device: Any = DEVICE,
    ) -> None:
        super(TimeSeriesTabularGAN, self).__init__()
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = TimeSeriesTabularEncoder(
                max_clusters=encoder_max_clusters
            ).fit(static_data, temporal_data)

        self.static_columns = static_data.columns
        self.temporal_columns = temporal_data[0].columns

        n_static_units, n_temporal_units = self.encoder.n_features()
        static_act, temporal_act = self.encoder.activation_layout(
            discrete_activation=generator_nonlin_out_discrete,
            continuous_activation=generator_nonlin_out_continuous,
        )
        self.model = TimeSeriesGAN(
            n_static_units=n_static_units,
            n_static_units_latent=n_static_units_latent,
            n_temporal_units=n_temporal_units,
            n_temporal_window=n_temporal_window,
            n_temporal_units_latent=n_temporal_units_latent,
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
            generator_opt_betas=generator_opt_betas,
            generator_extra_penalties=generator_extra_penalties,
            discriminator_n_units_hidden=discriminator_n_units_hidden,
            discriminator_n_layers_hidden=discriminator_n_layers_hidden,
            discriminator_n_iter=discriminator_n_iter,
            discriminator_nonlin=discriminator_nonlin,
            discriminator_batch_norm=discriminator_batch_norm,
            discriminator_dropout=discriminator_dropout,
            discriminator_loss=discriminator_loss,
            discriminator_lr=discriminator_lr,
            discriminator_weight_decay=discriminator_weight_decay,
            discriminator_extra_penalties=discriminator_extra_penalties,
            discriminator_opt_betas=discriminator_opt_betas,
            mode=mode,
            lambda_gradient_penalty=lambda_gradient_penalty,
            lambda_identifiability_penalty=lambda_identifiability_penalty,
            clipping_value=clipping_value,
            n_iter_print=n_iter_print,
            seed=seed,
            n_iter_min=n_iter_min,
            dataloader_sampler=dataloader_sampler,
            device=device,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def encode(
        self, static_data: pd.DataFrame, temporal_data: List[pd.DataFrame]
    ) -> Tuple:
        return self.encoder.transform(static_data, temporal_data)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def decode(
        self, static_data: pd.DataFrame, temporal_data: List[pd.DataFrame]
    ) -> Tuple:
        return self.encoder.inverse_transform(static_data, temporal_data)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        static_data: pd.DataFrame,
        temporal_data: List[pd.DataFrame],
        cond: Optional[Union[pd.DataFrame, pd.Series]] = None,
        fake_labels_generator: Optional[Callable] = None,
        true_labels_generator: Optional[Callable] = None,
        encoded: bool = False,
    ) -> Any:
        if encoded:
            static_enc = static_data
            temporal_enc = temporal_data
        else:
            static_enc, temporal_enc = self.encode(static_data, temporal_data)

        self.static_encoded_columns = static_enc.columns
        self.temporal_encoded_columns = temporal_enc[0].columns

        self.model.fit(
            np.asarray(static_enc),
            np.asarray(temporal_enc),
            np.asarray(cond),
            fake_labels_generator=fake_labels_generator,
            true_labels_generator=true_labels_generator,
        )
        return self

    def generate(self, count: int, cond: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        static_raw, temporal_raw = self.model.generate(count, cond)

        static_data = pd.DataFrame(static_raw, columns=self.static_encoded_columns)
        temporal_data = []
        for item in temporal_raw:
            temporal_data.append(
                pd.DataFrame(item, columns=self.temporal_encoded_columns)
            )

        return self.decode(static_data, temporal_data)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, count: int, cond: Optional[pd.DataFrame] = None) -> torch.Tensor:
        return self.model.forward(count, cond)
