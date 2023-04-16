# stdlib
from typing import Any, List, Optional, Tuple, Type

# third party
import numpy as np
import torch
from pydantic import validate_arguments
from torch import nn

# synthcity absolute
from synthcity.utils.constants import DEVICE

# synthcity relative
from .functions import EntmaxFunction, SparsemaxFunction


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


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def _forward_skip_connection(
    self: nn.Module, X: torch.Tensor, *args: Any, **kwargs: Any
) -> torch.Tensor:
    # if X.shape[-1] == 0:
    #     return torch.zeros((*X.shape[:-1], self.n_units_out)).to(self.device)
    X = X.float().to(self.device)
    out = self._forward(X, *args, **kwargs)
    return torch.cat([out, X], dim=-1)


def SkipConnection(cls: Type[nn.Module]) -> Type[nn.Module]:
    """Wraps a model to add a skip connection from the input to the output.

    Example:
    >>> ResidualBlock = SkipConnection(MLP)
    >>> res_block = ResidualBlock(n_units_in=10, n_units_out=3, n_units_hidden=64)
    >>> res_block(torch.ones(10, 10)).shape
    (10, 13)
    """

    class Wrapper(cls):  # type: ignore
        device: torch.device = DEVICE

    Wrapper._forward = cls.forward
    Wrapper.forward = _forward_skip_connection
    Wrapper.__name__ = f"SkipConnection({cls.__name__})"
    Wrapper.__qualname__ = f"SkipConnection({cls.__qualname__})"
    Wrapper.__doc__ = f"""(With skipped connection) {cls.__doc__}"""
    return Wrapper


# class GLU(nn.Module):
#     """Gated Linear Unit (GLU)."""

#     def __init__(self, activation: Union[str, nn.Module] = "sigmoid") -> None:
#         super().__init__()
#         if type(activation) == str:
#             self.non_lin = get_nonlin(activation)
#         else:
#             self.non_lin = activation

#     def forward(self, x: Tensor) -> Tensor:
#         if x.shape[-1] % 2:
#             raise ValueError("The last dimension of the input tensor must be even.")
#         a, b = x.chunk(2, dim=-1)
#         return a * self.non_lin(b)


class GumbelSoftmax(nn.Module):
    def __init__(
        self, tau: float = 0.2, hard: bool = False, eps: float = 1e-10, dim: int = -1
    ) -> None:
        super(GumbelSoftmax, self).__init__()

        self.tau = tau
        self.hard = hard
        self.eps = eps
        self.dim = dim

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.gumbel_softmax(
            logits, tau=self.tau, hard=self.hard, eps=self.eps, dim=self.dim
        )


class MultiActivationHead(nn.Module):
    """Final layer with multiple activations. Useful for tabular data."""

    def __init__(
        self,
        activations: List[Tuple[nn.Module, int]],
        device: Any = DEVICE,
    ) -> None:
        super(MultiActivationHead, self).__init__()
        self.activations = []
        self.activation_lengths = []
        self.device = device

        for activation, length in activations:
            self.activations.append(activation)
            self.activation_lengths.append(length)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[-1] != np.sum(self.activation_lengths):
            raise RuntimeError(
                f"Shape mismatch for the activations: expected {np.sum(self.activation_lengths)}. Got shape {X.shape}."
            )

        split = 0
        out = torch.zeros(X.shape).to(self.device)

        for activation, step in zip(self.activations, self.activation_lengths):
            out[..., split : split + step] = activation(X[..., split : split + step])

            split += step

        return out


class Sparsemax(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super(Sparsemax, self).__init__()
        self.dim = dim

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return SparsemaxFunction.apply(input, self.dim)


class Entmax(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super(Entmax, self).__init__()
        self.dim = dim

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return EntmaxFunction.apply(input, self.dim)
