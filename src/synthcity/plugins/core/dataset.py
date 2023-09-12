# stdlib
from typing import List, Optional, Tuple

# third party
import numpy as np
import torch

# synthcity absolute
from synthcity.utils.constants import DEVICE


class FlexibleDataset(torch.utils.data.Dataset):
    """Helper dataset wrapper for post-processing or transforming another dataset. Used for controlling the image sizes for the synthcity models.

    The class supports adding custom transforms to existing datasets, and to subsample a set of indices.

    Args:
        data: torch.Dataset
        transform: An optional list of transforms
        indices: An optional list of indices to subsample
    """

    def __init__(
        self,
        data: torch.utils.data.Dataset,
        transform: Optional[torch.nn.Module] = None,
        indices: Optional[list] = None,
    ) -> None:
        super().__init__()

        if indices is None:
            indices = np.arange(len(data))

        self.indices = np.asarray(indices)
        self.data = data
        self.transform = transform
        self.ndarrays: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.data[self.indices[index]]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.indices)

    def shape(self) -> Tuple:
        x, _ = self[self.indices[0]]

        return (len(self), *x.shape)

    def numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.ndarrays is not None:
            return self.ndarrays

        x_buff = []
        y_buff = []
        for idx in range(len(self)):
            x_local, y_local = self[idx]
            x_buff.append(x_local.unsqueeze(0).cpu().numpy())
            y_buff.append(y_local)

        x = np.concatenate(x_buff, axis=0)
        y = np.asarray(y_buff)

        self.ndarrays = (x, y)
        return x, y

    def tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.numpy()

        return torch.from_numpy(x), torch.from_numpy(y)

    def labels(self) -> np.ndarray:
        labels = []
        for idx in self.indices:
            _, y = self.data[idx]
            labels.append(y)

        return np.asarray(labels)

    def filter_indices(self, indices: List[int]) -> "FlexibleDataset":
        for idx in indices:
            if idx >= len(self.indices):
                raise ValueError(
                    "Invalid filtering list. {idx} not found in the current list of indices"
                )
        return FlexibleDataset(
            data=self.data, transform=self.transform, indices=self.indices[indices]
        )


class TensorDataset(torch.utils.data.Dataset):
    """Helper dataset for wrapping existing tensors

    Args:
        images: Tensor
        targets: Tensor
    """

    def __init__(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor],
    ) -> None:
        super().__init__()

        if targets is not None and len(targets) != len(images):
            raise ValueError("Invalid input")

        self.images = images
        self.targets = targets

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y: Optional[torch.Tensor] = None
        x = self.images[index]

        if self.targets is not None:
            y = self.targets[index]

        return x, y

    def __len__(self) -> int:
        return len(self.images)

    def labels(self) -> Optional[np.ndarray]:
        if self.targets is None:
            return None

        return self.targets.cpu().numpy()


class ConditionalDataset(torch.utils.data.Dataset):
    """Helper dataset for wrapping existing datasets with custom tensors

    Args:
        data: torch.Dataset
        cond: Optional Tensor
    """

    def __init__(
        self,
        data: torch.utils.data.Dataset,
        cond: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        if cond is not None and len(cond) != len(data):
            raise ValueError("Invalid input")

        self.data = data
        self.cond = cond

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cond: Optional[torch.Tensor] = None
        x = self.data[index][0]

        if self.cond is not None:
            cond = self.cond[index]

        return x, cond

    def __len__(self) -> int:
        return len(self.data)


class NumpyDataset(torch.utils.data.Dataset):
    """Helper class for wrapping Numpy arrays in torch Datasets
    Args:
        X: np.ndarray
        y: np.ndarray
    """

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
