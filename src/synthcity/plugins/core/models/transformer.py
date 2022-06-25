# stdlib
from typing import Any, Optional

# third party
import torch
from torch import nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

# synthcity absolute
from synthcity.utils.constants import DEVICE


class Permute(nn.Module):
    def __init__(self, *dims: Any) -> None:
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self.dims)


class Max(nn.Module):
    def __init__(self, dim: Optional[int] = None, keepdim: bool = False):
        super(Max, self).__init__()
        self.dim, self.keepdim = dim, keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(self.dim, keepdim=self.keepdim)[0]


class Transpose(nn.Module):
    def __init__(self, *dims: Any, contiguous: bool = False) -> None:
        super(Transpose, self).__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class TransformerModel(nn.Module):
    def __init__(
        self,
        n_units_in: int,
        n_units_hidden: int = 64,
        n_head: int = 1,
        d_ffn: int = 128,
        dropout: float = 0.1,
        activation: str = "relu",
        n_hidden_layers: int = 1,
        device: Any = DEVICE,
    ) -> None:
        """
        Args:
            c_in: the number of features (aka variables, dimensions, channels) in the time series dataset
            c_out: the number of target classes
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
        self.permute = Permute(1, 0, 2)
        self.inlinear = nn.Linear(n_units_in, n_units_hidden)
        self.relu = nn.ReLU()
        encoder_layer = TransformerEncoderLayer(
            n_units_hidden,
            n_head,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation=activation,
        )
        encoder_norm = nn.LayerNorm(n_units_hidden)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, n_hidden_layers, norm=encoder_norm
        )
        self.transpose = Transpose(1, 0)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.permute(x)  # bs x seq_len x nvars -> seq_len x bs x nvars
        x = self.inlinear(x)  # seq_len x bs x nvars -> seq_len x bs x n_units_hidden
        x = self.relu(x)
        x = self.transformer_encoder(x)
        x = self.transpose(
            x
        )  # seq_len x bs x n_units_hidden -> bs x seq_len x n_units_hidden
        x = self.relu(x)
        return x
