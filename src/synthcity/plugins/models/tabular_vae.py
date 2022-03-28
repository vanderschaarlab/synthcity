# stdlib
from typing import Any

# third party
import pandas as pd
import torch
from pydantic import validate_arguments
from torch import nn

# synthcity relative
from .tabular_encoder import TabularEncoder
from .vae import VAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabularVAE(nn.Module):
    """
    Tabular VAE implementation.

    Args:
        n_units_in: int
            Number of features
        decoder_n_layers_hidden: int
            Number of hidden layers in the decoder
        decoder_n_units_hidden: int
            Number of hidden units in each layer of the decoder
        decoder_nonlin: string, default 'elu'
            Nonlinearity to use in the decoder. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        decoder_n_iter: int
            Maximum number of iterations in the decoder.
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
        seed: int
            Seed used
        n_iter_min: int
            Minimum number of iterations to go through before starting early stopping
        clipping_value: int, default 1
            Gradients clipping value
        encoder_max_clusters: int
            The max number of clusters to create for continuous columns when encoding
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        X: pd.DataFrame,
        n_units_embedding: int,
        lr: float = 2e-4,
        n_iter: int = 500,
        weight_decay: float = 1e-3,
        batch_size: int = 64,
        n_iter_print: int = 10,
        seed: int = 0,
        clipping_value: int = 1,
        loss_strategy: str = "standard",
        encoder_max_clusters: int = 20,
        decoder_n_layers_hidden: int = 2,
        decoder_n_units_hidden: int = 250,
        decoder_nonlin: str = "leaky_relu",
        decoder_nonlin_out_discrete: str = "softmax",
        decoder_nonlin_out_continuous: str = "tanh",
        decoder_batch_norm: bool = False,
        decoder_dropout: float = 0,
        decoder_residual: bool = True,
        encoder_n_layers_hidden: int = 3,
        encoder_n_units_hidden: int = 300,
        encoder_nonlin: str = "leaky_relu",
        encoder_batch_norm: bool = False,
        encoder_dropout: float = 0.1,
    ) -> None:
        super(TabularVAE, self).__init__()
        self.columns = X.columns
        self.encoder = TabularEncoder(max_clusters=encoder_max_clusters).fit(X)

        self.model = VAE(
            self.encoder.n_features(),
            n_units_embedding=n_units_embedding,
            batch_size=batch_size,
            n_iter=n_iter,
            lr=lr,
            weight_decay=weight_decay,
            clipping_value=clipping_value,
            seed=seed,
            n_iter_print=n_iter_print,
            loss_strategy=loss_strategy,
            decoder_n_layers_hidden=decoder_n_layers_hidden,
            decoder_n_units_hidden=decoder_n_units_hidden,
            decoder_nonlin=decoder_nonlin,
            decoder_nonlin_out=self.encoder.activation_layout(
                discrete_activation=decoder_nonlin_out_discrete,
                continuous_activation=decoder_nonlin_out_continuous,
            ),
            decoder_batch_norm=decoder_batch_norm,
            decoder_dropout=decoder_dropout,
            decoder_residual=decoder_residual,
            encoder_n_units_hidden=encoder_n_units_hidden,
            encoder_n_layers_hidden=encoder_n_layers_hidden,
            encoder_nonlin=encoder_nonlin,
            encoder_batch_norm=encoder_batch_norm,
            encoder_dropout=encoder_dropout,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def encode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.transform(X)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def decode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.inverse_transform(X)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        X: pd.DataFrame,
    ) -> Any:
        X_enc = self.encode(X)
        self.model.fit(
            X_enc,
        )
        return self

    def generate(self, count: int) -> pd.DataFrame:
        samples = self.model.generate(count)
        return self.decode(pd.DataFrame(samples))

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, count: int) -> torch.Tensor:
        return self.model.forward(count)
