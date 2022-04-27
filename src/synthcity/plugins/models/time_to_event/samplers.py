# stdlib
from typing import Generator

# third party
import pandas as pd
import torch
import torch.utils.data


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset"""

    def __init__(self, X: torch.Tensor, T: torch.Tensor, E: torch.Tensor) -> None:
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(X.shape[0]))

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = E.cpu().numpy()
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def __iter__(self) -> Generator:
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self) -> int:
        return self.num_samples
