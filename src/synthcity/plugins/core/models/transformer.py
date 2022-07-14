# stdlib
from typing import Any

# third party
import torch
from torch import nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

# synthcity absolute
from synthcity.utils.constants import DEVICE

# synthcity relative
from .layers import Permute, Transpose


class TransformerModel(nn.Module):
    def __init__(
        self,
        n_units_in: int,
        n_units_hidden: int = 64,
        n_head: int = 1,
        d_ffn: int = 128,
        dropout: float = 0.1,
        activation: str = "relu",
        n_layers_hidden: int = 1,
        device: Any = DEVICE,
    ) -> None:
        """
        Args:
            c_in: the number of features (aka variables, dimensions, channels) in the time series dataset
            n_units_hidden: total dimension of the model.
            nhead:  parallel attention heads.
            d_ffn: the dimension of the feedforward network model.
            dropout: a Dropout layer on attn_output_weights.
            activation: the activation function of intermediate layer, relu or gelu.
            num_layers: the number of sub-encoder-layers in the encoder.

        Input shape:
            bs (batch size) x seq_len (aka time steps) x nvars (aka variables, dimensions, channels)
        """
        super(TransformerModel, self).__init__()
        encoder_layer = TransformerEncoderLayer(
            n_units_hidden,
            n_head,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation=activation,
        )
        encoder_norm = nn.LayerNorm(n_units_hidden)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, n_layers_hidden, norm=encoder_norm
        )
        self.model = nn.Sequential(
            Permute(1, 0, 2),  # bs x seq_len x nvars -> seq_len x bs x nvars
            nn.Linear(
                n_units_in, n_units_hidden
            ),  # seq_len x bs x nvars -> seq_len x bs x n_units_hidden
            nn.ReLU(),
            TransformerEncoderLayer(
                n_units_hidden,
                n_head,
                dim_feedforward=d_ffn,
                dropout=dropout,
                activation=activation,
            ),
            Transpose(
                1, 0
            ),  # seq_len x bs x n_units_hidden -> bs x seq_len x n_units_hidden
            nn.ReLU(),
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
