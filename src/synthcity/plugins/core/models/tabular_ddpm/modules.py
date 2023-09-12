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
from synthcity.plugins.core.models.factory import get_model, get_nonlin


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
        fs = fs.to(timesteps.device)
        args = timesteps[:, None].float() * fs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.fc(emb)


class DiffusionModel(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_emb: int = 128,
        *,
        model_type: str = "mlp",
        model_params: dict = {},
        conditional: bool = False,
        num_classes: int = 0,
        emb_nonlin: Union[str, nn.Module] = "silu",
        max_time_period: int = 10000,
    ) -> None:
        super().__init__()
        self.dim_t = dim_emb
        self.num_classes = num_classes
        self.has_label = conditional

        if isinstance(emb_nonlin, str):
            self.emb_nonlin = get_nonlin(emb_nonlin)
        else:
            self.emb_nonlin = emb_nonlin

        self.proj = nn.Linear(dim_in, dim_emb)
        self.time_emb = TimeStepEmbedding(dim_emb, max_time_period)

        if conditional:
            if self.num_classes > 0:
                self.label_emb = nn.Embedding(self.num_classes, dim_emb)
            elif self.num_classes == 0:  # regression
                self.label_emb = nn.Linear(1, dim_emb)

        if not model_params:
            model_params = {}  # avoid changing the default dict

        if model_type == "mlp":
            if not model_params:
                model_params = dict(n_units_hidden=256, n_layers_hidden=3, dropout=0.0)
            model_params.update(n_units_in=dim_emb, n_units_out=dim_in)
        elif model_type == "tabnet":
            model_params.update(input_dim=dim_emb, output_dim=dim_in)

        self.model = get_model(model_type, model_params)

    def forward(self, x: Tensor, t: Tensor, y: Optional[Tensor] = None) -> Tensor:
        emb = self.time_emb(t)
        if self.has_label:
            if y is None:
                raise ValueError("y must be provided if conditional is True")
            if self.num_classes == 0:
                y = y.reshape(-1, 1).float()
            else:
                y = y.squeeze().long()
            emb += self.emb_nonlin(self.label_emb(y))
        x = self.proj(x) + emb
        return self.model(x)
