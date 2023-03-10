# flake8: noqa: F401

# stdlib
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# third party
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def normal_kl(mean1: Tensor, logvar1: Tensor, mean2: Tensor, logvar2: Tensor) -> Tensor:
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, Tensor):
            tensor = obj
            break
    if tensor is None:
        raise AssertionError("at least one argument must be a Tensor")

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x: Tensor) -> Tensor:
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (
        1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
    )


def discretized_gaussian_log_likelihood(
    x: Tensor, *, means: Tensor, log_scales: Tensor
) -> Tensor:
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    if not (x.shape == means.shape == log_scales.shape):
        raise AssertionError
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))
        ),
    )
    if not (log_probs.shape == x.shape):
        raise AssertionError
    return log_probs


def sum_except_batch(x: Tensor, num_dims: int = 1) -> Tensor:
    """
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    """
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def mean_flat(tensor: Tensor) -> Tensor:
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def ohe_to_categories(ohe: Tensor, K: np.ndarray) -> Tensor:
    K = torch.from_numpy(K)
    indices = torch.cat([torch.zeros((1,)), K.cumsum(dim=0)], dim=0).int().tolist()
    res = []
    for i in range(len(indices) - 1):
        res.append(ohe[:, indices[i] : indices[i + 1]].argmax(dim=1))
    return torch.stack(res, dim=1)


def log_1_min_a(a: Tensor) -> Tensor:
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a: Tensor, b: Tensor) -> Tensor:
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(a: Tensor, t: Tensor, x_shape: tuple) -> Tensor:
    b, *_ = t.shape
    t = t.to(a.device)
    out = a.gather(-1, t)
    while len(out.shape) < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape)


def log_categorical(log_x_start: Tensor, log_prob: Tensor) -> Tensor:
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x: Tensor, num_classes: np.ndarray) -> Tensor:
    onehots = []
    for i in range(len(num_classes)):
        onehots.append(F.one_hot(x[:, i], num_classes[i]))
    x_onehot = torch.cat(onehots, dim=1)
    log_onehot = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_onehot


@torch.jit.script
def log_sub_exp(a: Tensor, b: Tensor) -> Tensor:
    m = torch.maximum(a, b)
    return torch.log(torch.exp(a - m) - torch.exp(b - m)) + m


@torch.jit.script
def sliced_logsumexp(x: Tensor, slices: Tensor) -> Tensor:
    lse = torch.logcumsumexp(
        torch.nn.functional.pad(x, [1, 0, 0, 0], value=-float("inf")), dim=-1
    )

    slice_starts = slices[:-1]
    slice_ends = slices[1:]

    slice_lse = log_sub_exp(lse[:, slice_ends], lse[:, slice_starts])
    slice_lse_repeated = torch.repeat_interleave(
        slice_lse, slice_ends - slice_starts, dim=-1
    )
    return slice_lse_repeated


class FoundNANsError(BaseException):
    """Found NANs during sampling"""

    def __init__(self, message: str = "Found NANs during sampling.") -> None:
        super(FoundNANsError, self).__init__(message)


class TensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(
        self, *tensors: Tensor, batch_size: int = 32, shuffle: bool = False
    ) -> None:
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        if not all(t.shape[0] == tensors[0].shape[0] for t in tensors):
            raise AssertionError
        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[tuple]:
        idx = np.arange(self.dataset_len)
        if self.shuffle:
            np.random.shuffle(idx)
        for i in range(0, self.dataset_len, self.batch_size):
            s = idx[i : i + self.batch_size]
            yield tuple(t[s] for t in self.tensors)

    def __len__(self) -> int:
        return len(range(0, self.dataset_len, self.batch_size))
