# stdlib
import math
from typing import Callable, List

# third party
import numpy as np
import scipy.optimize as sopt
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def has_cycle(W: np.ndarray, v: int, visited: List[bool], recstack: List[bool]) -> bool:
    visited[v] = True
    recstack[v] = True

    for neighbour in range(len(W[v])):
        if W[v][neighbour] == 0:
            continue
        if not visited[neighbour]:
            if has_cycle(W, neighbour, visited, recstack):
                return True
        elif recstack[neighbour]:
            return True

    recstack[v] = False
    return False


def is_dag(W: np.ndarray) -> bool:
    visited = [False] * len(W)
    recstack = [False] * len(W)

    for node in range(len(W)):
        if not visited[node]:
            if has_cycle(W, node, visited, recstack):
                return False
    return True


# Zheng et al.
class LBFGSBScipy(torch.optim.Optimizer):
    """Wrap L-BFGS-B algorithm, using scipy routines."""

    def __init__(self, params: dict) -> None:
        super(LBFGSBScipy, self).__init__(params, {})

        if len(self.param_groups) != 1:
            raise ValueError(
                "LBFGSBScipy doesn't support per-parameter options"
                " (parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._numel = sum([p.numel() for p in self._params])

    def _gather_flat_grad(self) -> torch.Tensor:
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0).to(DEVICE)

    def _gather_flat_bounds(self) -> list:
        bounds = []
        for p in self._params:
            if hasattr(p, "bounds"):
                b = p.bounds
            else:
                b = [(None, None)] * p.numel()
            bounds += b
        return bounds

    def _gather_flat_params(self) -> torch.Tensor:
        views = []
        for p in self._params:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0).to(DEVICE)

    def _distribute_flat_params(self, params: dict) -> None:
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data = params[offset : offset + numel].view_as(p.data)
            offset += numel
        if offset != self._numel:
            raise RuntimeError(f"Invalid offset = {offset}")

    def step(self, closure: Callable) -> None:
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        if len(self.param_groups) != 1:
            raise RuntimeError(f"Invalid param groups {self.param_groups}")

        def wrapped_closure(flat_params: torch.Tensor) -> tuple:
            """closure must call zero_grad() and backward()"""
            flat_params = torch.from_numpy(flat_params).to(DEVICE).double()
            self._distribute_flat_params(flat_params)
            loss = closure()
            loss = loss.item()
            flat_grad = self._gather_flat_grad().cpu().detach().numpy()
            return loss, flat_grad.astype("float64")

        initial_params = self._gather_flat_params()
        initial_params = initial_params.cpu().detach().numpy()

        bounds = self._gather_flat_bounds()

        # Magic
        sol = sopt.minimize(
            wrapped_closure, initial_params, method="L-BFGS-B", jac=True, bounds=bounds
        )

        final_params = torch.from_numpy(sol.x).to(DEVICE).double()

        self._distribute_flat_params(final_params)


# Zheng et al.
class LocallyConnected(nn.Module):
    """Local linear layer, i.e. Conv1dLocal() with filter size 1.
    Args:
        num_linear: num of local linear layers, i.e.
        in_features: m1
        out_features: m2
        bias: whether to include bias or not
    Shape:
        - Input: [n, d, m1]
        - Output: [n, d, m2]
    Attributes:
        weight: [d, m1, m2]
        bias: [d, m2]
    """

    def __init__(
        self,
        num_linear: int,
        input_features: int,
        output_features: int,
        bias: bool = True,
    ) -> None:
        super(LocallyConnected, self).__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(
            torch.Tensor(num_linear, input_features, output_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter("bias", None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = torch.matmul(input.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self) -> str:
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return "num_linear={}, in_features={}, out_features={}, bias={}".format(
            self.num_linear,
            self.input_features,
            self.output_features,
            self.bias is not None,
        )


# Zheng et al.
def count_accuracy(B_true: np.ndarray, B_est: np.ndarray) -> dict:
    """Compute various accuracy metrics for B_est.
    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition
    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG
    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError("B_est should take value in {0,1,-1}")
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError("undirected edge should only appear once")
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError("B_est should take value in {0,1}")
        if not is_dag(B_est):
            raise ValueError("B_est should be a DAG")
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {"fdr": fdr, "tpr": tpr, "fpr": fpr, "shd": shd, "nnz": pred_size}
