# stdlib
from typing import Generator, List

# third party
import pandas as pd
import torch
import torch.utils.data
from pydantic import validate_arguments


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset"""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(self, labels: List) -> None:
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(labels)))

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]
        print("using sampling weights", weights)

        self.weights = torch.DoubleTensor(weights.to_list())

    def __iter__(self) -> Generator:
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self) -> int:
        return self.num_samples
