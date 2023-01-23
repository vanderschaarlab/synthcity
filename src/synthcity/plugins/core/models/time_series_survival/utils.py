# stdlib
from typing import Any, Optional, Tuple

# third party
import numpy as np
from sklearn.utils import check_consistent_length


def get_padded_features(
    x: np.ndarray, pad_size: Optional[int] = None, fill: int = np.nan
) -> np.ndarray:
    """Helper function to pad variable length RNN inputs with nans."""
    if pad_size is None:
        pad_size = max([len(x_) for x_ in x])

    padx = []
    for i in range(len(x)):
        if pad_size == len(x[i]):
            padx.append(x[i].astype(float))
        elif pad_size > len(x[i]):
            pads = fill * np.ones((pad_size - len(x[i]),) + x[i].shape[1:])
            padx.append(np.concatenate([x[i], pads]).astype(float))
        else:
            padx.append(x[i][:pad_size].astype(float))

    return np.asarray(padx)


class StepFunction:
    """Callable step function.

    .. math::

        f(z) = a * y_i + b,
        x_i \\leq z < x_{i + 1}

    Parameters
    ----------
    x : ndarray, shape = (n_points,)
        Values on the x axis in ascending order.

    y : ndarray, shape = (n_points,)
        Corresponding values on the y axis.

    a : float, optional, default: 1.0
        Constant to multiply by.

    b : float, optional, default: 0.0
        Constant offset term.
    """

    def __init__(
        self, x: np.ndarray, y: np.ndarray, a: float = 1.0, b: float = 0.0
    ) -> None:
        check_consistent_length(x, y)
        self.x = x
        self.y = y
        self.a = a
        self.b = b

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate step function.

        Parameters
        ----------
        x : float|array-like, shape=(n_values,)
            Values to evaluate step function at.

        Returns
        -------
        y : float|array-like, shape=(n_values,)
            Values of step function at `x`.
        """
        x = np.atleast_1d(x)
        if not np.isfinite(x).all():
            raise ValueError("x must be finite")
        if np.min(x) < self.x[0] or np.max(x) > self.x[-1]:
            raise ValueError("x must be within [%f; %f]" % (self.x[0], self.x[-1]))
        i = np.searchsorted(self.x, x, side="left")
        not_exact = self.x[i] != x
        i[not_exact] -= 1
        value = self.a * self.y[i] + self.b
        if value.shape[0] == 1:
            return value[0]
        return value

    def __repr__(self) -> str:
        return "StepFunction(x=%r, y=%r, a=%r, b=%r)" % (self.x, self.y, self.a, self.b)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, type(self)):
            return (
                all(self.x == other.x)
                and all(self.y == other.y)
                and self.a == other.a
                and self.b == other.b
            )
        return False


def _compute_counts(
    event: np.ndarray, time: np.ndarray, order: Optional[np.ndarray] = None
) -> Tuple:
    """Count right censored and uncensored samples at each unique time point.

    Parameters
    ----------
    event : array
        Boolean event indicator.

    time : array
        Survival time or time of censoring.

    order : array or None
        Indices to order time in ascending order.
        If None, order will be computed.

    Returns
    -------
    times : array
        Unique time points.

    n_events : array
        Number of events at each time point.

    n_at_risk : array
        Number of samples that have not been censored or have not had an event at each time point.

    n_censored : array
        Number of censored samples at each time point.
    """
    n_samples = event.shape[0]

    if order is None:
        order = np.argsort(time, kind="mergesort")

    uniq_times = np.empty(n_samples, dtype=time.dtype)
    uniq_events = np.empty(n_samples, dtype=int)
    uniq_counts = np.empty(n_samples, dtype=int)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    times = np.resize(uniq_times, j)
    n_events = np.resize(uniq_events, j)
    total_count = np.resize(uniq_counts, j)
    n_censored = total_count - n_events

    # offset cumulative sum by one
    total_count = np.r_[0, total_count]
    n_at_risk = n_samples - np.cumsum(total_count)

    return times, n_events, n_at_risk[:-1], n_censored
