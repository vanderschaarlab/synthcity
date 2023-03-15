"""
Code was adapted from https://github.com/Yura52/rtdl
"""
# stdlib
import math
from typing import Optional, Union

# third party
import torch
import torch.optim
from torch import Tensor, nn

# synthcity absolute
from synthcity.plugins.core.models.mlp import MLP, get_nonlin


class TimeStepEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_period: int = 10000,
        n_layers: int = 2,
        nonlin: Union[str, nn.Module] = "silu",
    ) -> None:
        """
        Create sinusoidal timestep embeddings.

        Args:
        - dim (int): the dimension of the output.
        - max_period (int): controls the minimum frequency of the embeddings.
        - n_layers (int): number of dense layers
        """
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.n_layers = n_layers

        if dim % 2 != 0:
            raise ValueError(f"embedding dim must be even, got {dim}")

        layers = []
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(dim, dim))
            layers.append(get_nonlin(nonlin))

        self.fc = nn.Sequential(*layers, nn.Linear(dim, dim))

    def forward(self, timesteps: Tensor) -> Tensor:
        """
        Args:
        - timesteps (Tensor): 1D Tensor of N indices, one per batch element.
        """
        d, T = self.dim, self.max_period
        mid = d // 2
        fs = torch.exp(-math.log(T) / mid * torch.arange(mid, dtype=torch.float32))
        args = timesteps[:, None].float() * fs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.fc(emb)


class MLPDiffusion(nn.Module):
    add_residual = False

    def __init__(
        self,
        dim_in: int,
        dim_emb: int = 128,
        *,
        mlp_params: dict = {},
        use_label: bool = False,
        num_classes: int = 0,
        emb_nonlin: Union[str, nn.Module] = "silu",
        max_time_period: int = 10000,
    ) -> None:
        super().__init__()
        self.dim_t = dim_emb
        self.num_classes = num_classes
        self.has_label = use_label

        if isinstance(emb_nonlin, str):
            self.emb_nonlin = get_nonlin(emb_nonlin)
        else:
            self.emb_nonlin = emb_nonlin

        self.proj = nn.Linear(dim_in, dim_emb)
        self.time_emb = TimeStepEmbedding(dim_emb, max_time_period)

        if use_label:
            if self.num_classes > 0:
                self.label_emb = nn.Embedding(self.num_classes, dim_emb)
            elif self.num_classes == 0:  # regression
                self.label_emb = nn.Linear(1, dim_emb)

        self.model = MLP(
            n_units_in=dim_emb,
            n_units_out=dim_in,
            task_type="/",
            residual=self.add_residual,
            **mlp_params,
        )

    def forward(self, x: Tensor, t: Tensor, y: Optional[Tensor] = None) -> Tensor:
        emb = self.time_emb(t)
        if self.has_label:
            if y is None:
                raise ValueError("y must be provided if use_label is True")
            if self.num_classes == 0:
                y = y.resize(-1, 1).float()
            else:
                y = y.squeeze().long()
            emb += self.emb_nonlin(self.label_emb(y))
        x = self.proj(x) + emb
        return self.model(x)


class ResNetDiffusion(MLPDiffusion):
    add_residual = True
