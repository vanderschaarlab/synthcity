# stdlib
import os
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional

# third party
import numpy as np
import pytorch_lightning as pl
import scipy.stats as stats
import synthicty.plugins.core.models.dag.simulate as sm
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class Data(pl.LightningDataModule):
    def __init__(
        self,
        dim: int = 20,  # Amount of vars
        s0: int = 40,  # Expected amount of edges
        N: int = 1000,  # Amount of samples
        sem_type: str = "mim",  # SEM-type (
        #   'mim' -> index model,
        #   'mlp' -> multi layer perceptrion,
        #   'gp' -> gaussian process,
        #   'gp-add' -> addative gp)
        dag_type: str = "ER",  # Random graph type (
        #   'ER' -> Erdos-Renyi,
        #   'SF' -> Scale Free,
        #   'BP' -> BiPartite)
        batch_size: int = 32,
        train_size_ratio: float = 1.0,  # Ratio of train/test split
        loc: str = "data/simulated/",  # Save location for simulated data
        sort: bool = False,  # sort covariates
        rand_sort: bool = False,  # random sort (permute) covariates
    ):
        super().__init__()

        self.dim = dim
        self.s0 = s0
        self.N = N
        self.sem_type = sem_type
        self.dag_type = dag_type

        self.batch_size = batch_size
        self.train_size_ratio = train_size_ratio

        self.loc = Path(loc)

        self.DAG: Optional[np.ndarray] = None
        self._simulate()
        self._sample()

    def _simulate(self) -> None:
        self.DAG = sm.simulate_dag(self.dim, self.s0, self.dag_type)
        self._id = hash(
            self.DAG.__repr__() + self.DAG.__array_interface__["data"][0].__repr__()
        )

        path = self.loc / str(self._id)

        os.makedirs(path, exist_ok=True)

        np.savetxt(path / "DAG.csv", self.DAG, delimiter=",")

    def _sample(self) -> None:
        assert self.DAG is not None, "No DAG simulated yet"

        self.X = sm.simulate_nonlinear_sem(self.DAG, self.N, self.sem_type)

        np.savetxt(self.loc / str(self._id) / "X.csv", self.X, delimiter=",")

    def setup(self, stage: Optional[str] = None) -> None:
        assert self.DAG is not None, "No DAG simulated yet"
        assert self.X is not None, "No SEM simulated yet"

        DX = TensorDataset(torch.from_numpy(self.X))

        self._train_size = int(np.floor(self.N * self.train_size_ratio))
        self.train, self.test = random_split(
            DX, [int(self._train_size), int(self.N - self._train_size)]
        )

    def resample(self) -> None:
        """
        Resamples a new DAG and SEM
        Resets the train and test sets
        Writes new data and DAG to self.loc
            without overwriting previous data
        """
        self._simulate()
        self._sample()
        self.setup()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=os.cpu_count()
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test, batch_size=len(self.test), num_workers=os.cpu_count()
        )


class Subset(pl.LightningDataModule):
    def __init__(
        self, X: np.ndarray, train_size_ratio: float = 0.5, batch_size: int = 256
    ) -> None:
        super().__init__()

        self.train_size_ratio = train_size_ratio
        self.batch_size = batch_size

        self.X = X
        self.N = self.X.shape[0]

    def setup(self) -> None:
        DX = TensorDataset(torch.from_numpy(self.X))

        _train_size = np.floor(self.N * self.train_size_ratio)
        self.train, self.test = random_split(
            DX, [int(_train_size), int(self.N - _train_size)]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=os.cpu_count()
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=os.cpu_count()
        )


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
        assert K > 0, f"Cannot have K of {K}"

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
