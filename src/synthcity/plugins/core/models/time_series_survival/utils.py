# stdlib
from typing import Optional

# third party
import numpy as np


def get_padded_features(
    x: np.ndarray, pad_size: Optional[int] = None, fill: int = np.nan
) -> np.ndarray:
    """Helper function to pad variable length RNN inputs with nans."""
    if pad_size is None:
        pad_size = max([len(x_) for x_ in x])

    padx = []
    for i in range(len(x)):
        if pad_size == len(x[i]):
            padx.append(x[i])
        elif pad_size > len(x[i]):
            pads = fill * np.ones((pad_size - len(x[i]),) + x[i].shape[1:])
            padx.append(np.concatenate([x[i], pads]))
        else:
            padx.append(x[i][:pad_size])

    return np.array(padx)
