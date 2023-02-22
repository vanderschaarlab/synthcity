# stdlib
from typing import Tuple

# third party
import numpy as np
import torch

# synthcity absolute
from synthcity.utils.constants import DEVICE


class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        super().__init__()

        self.X = X
        self.y = y

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[index]
        y = self.y[index]

        return torch.from_numpy(x).to(DEVICE), y

    def __len__(self) -> int:
        return len(self.X)
