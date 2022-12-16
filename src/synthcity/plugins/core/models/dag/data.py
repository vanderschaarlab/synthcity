# stdlib
from abc import abstractmethod
from typing import List, Optional

# third party
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader, TensorDataset


class P:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, batch: torch.Tensor) -> List[torch.Tensor]:
        pass


class BetaP(P):
    def __init__(self, K: int, sort: bool = False, rand_sort: bool = False) -> None:
        super().__init__()

        self.betas = self._get_betas(K)
        self.sort = sort
        self.rand_sort = rand_sort

    def _get_betas(self, K: int) -> list:
        if K == 0:
            raise RuntimeError(f"Cannot have K of {K}")

        first_half = [(i, K) for i in np.linspace(1, K - 1, int((K - K % 2) / 2))]
        second_half = [(K, i) for i in np.linspace(K - 1, 1, int((K - K % 2) / 2))]
        mid = [(K, K)] if K % 2 else []

        params = [*first_half, *mid, *second_half]
        betas = [stats.beta(*param) for param in params]

        return betas

    def __call__(self, batch: torch.Tensor) -> List[torch.Tensor]:
        N = batch.shape[0]

        if self.sort:
            batch, _ = torch.sort(batch, dim=1)

        if self.rand_sort:
            batch = batch[torch.randperm(batch.size()[0])]

        subsets = []
        for beta in self.betas:
            probs = beta.pdf(np.linspace(0, 1, N))
            probs = (probs - probs.min()) / (probs.max() - probs.min())

            mask = np.random.binomial(1, probs)

            X = batch[mask == 1]
            subsets.append(X)
        return subsets


class CustomDataModule(pl.LightningDataModule):
    def __init__(
        self, X: pd.DataFrame, batch_size: int = 256, train_size: Optional[float] = None
    ):
        super().__init__()

        self.batch_size = batch_size

        X = np.asarray(X)
        self.X = TensorDataset(torch.from_numpy(X))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.X, batch_size=self.batch_size)
