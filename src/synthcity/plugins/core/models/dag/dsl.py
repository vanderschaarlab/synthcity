# stdlib
from typing import Any, Optional, Tuple

# third party
import numpy as np
import torch
import torch.nn as nn

# synthcity absolute
from synthcity.plugins.core.models.dag.utils import LocallyConnected

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, data: torch.Tensor) -> torch.Tensor:
        E = torch.linalg.matrix_exp(data)
        f = torch.trace(E)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=data.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (E,) = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input


trace_expm = TraceExpm.apply


# From Zheng et al. (2020)
class NotearsMLP(nn.Module):
    def __init__(self, dims: list, bias: bool = True, priors: np.ndarray = []):
        super(NotearsMLP, self).__init__()
        if len(dims) < 2:
            raise ValueError(f"Invalid dims = {dims}")
        if dims[-1] != 1:
            raise ValueError(f"Invalid dims[-1] = {dims[-1]}")

        d = dims[0]
        self.dims = dims
        self.priors = priors

        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias).to(DEVICE).double()
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias).to(DEVICE).double()
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for layer in range(len(dims) - 2):
            layers.append(
                LocallyConnected(d, dims[layer + 1], dims[layer + 2], bias=bias)
            )
        self.fc2 = nn.ModuleList(layers).to(DEVICE).double()

    def _check(self, target: Any) -> bool:
        if len(self.priors) != 0:
            return any((self.priors[:, None] == target).all(2).any(1))
        else:
            return False

    def _bounds(self) -> list:
        d = self.dims[0]
        bounds = []
        for j in range(d):
            bound: Tuple[int, Optional[int]]
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    elif self._check([i, j]):
                        bound = (0, 0)
                    elif self._check([j, i]):
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self) -> torch.Tensor:
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        h = trace_expm(A) - d  # (Zheng et al. 2018)
        return h

    def l2_reg(self) -> torch.Tensor:
        """Take 2-norm-squared of all parameters"""
        reg = 0.0
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight**2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight**2)
        return reg

    def fc1_l1_reg(self) -> torch.Tensor:
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    def fc1_to_adj_grad(self) -> torch.Tensor:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        return W

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:
        W = self.fc1_to_adj_grad()
        return W.cpu().detach().numpy()


class NotearsSobolev(nn.Module):
    def __init__(self, d: int, k: int) -> None:
        """d: num variables k: num expansion of each variable"""
        super(NotearsSobolev, self).__init__()
        self.d, self.k = d, k
        self.fc1_pos = nn.Linear(d * k, d, bias=False)  # ik -> j
        self.fc1_neg = nn.Linear(d * k, d, bias=False)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        nn.init.zeros_(self.fc1_pos.weight)
        nn.init.zeros_(self.fc1_neg.weight)
        self.l2_reg_store = None

    def _bounds(self) -> list:
        # weight shape [j, ik]
        bounds = []
        for j in range(self.d):
            for i in range(self.d):
                bound: Tuple[int, Optional[int]]
                for _ in range(self.k):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def sobolev_basis(self, x: torch.Tensor) -> torch.Tensor:  # [n, d] -> [n, dk]
        seq = []
        for kk in range(self.k):
            mu = 2.0 / (2 * kk + 1) / np.pi  # sobolev basis
            psi = mu * torch.sin(x / mu)
            seq.append(psi)  # [n, d] * k
        bases = torch.stack(seq, dim=2)  # [n, d, k]
        bases = bases.view(-1, self.d * self.k)  # [n, dk]
        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n, d] -> [n, d]
        bases = self.sobolev_basis(x)  # [n, dk]
        x = self.fc1_pos(bases) - self.fc1_neg(bases)  # [n, d]
        self.l2_reg_store = torch.sum(x**2) / x.shape[0]
        return x

    def h_func(self) -> torch.Tensor:
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j, ik]
        fc1_weight = fc1_weight.view(self.d, self.d, self.k)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        h = trace_expm(A) - self.d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(self.d) + A / self.d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, self.d - 1)
        # h = (E.t() * M).sum() - self.d
        return h

    def l2_reg(self) -> torch.Tensor:
        reg = self.l2_reg_store
        return reg

    def fc1_l1_reg(self) -> torch.Tensor:
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    def fc1_to_adj_grad(self) -> torch.Tensor:
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j, ik]
        fc1_weight = fc1_weight.view(self.d, self.d, self.k)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]

        return W

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:
        W = self.fc1_to_adj_grad()
        return W.cpu().detach().numpy()
