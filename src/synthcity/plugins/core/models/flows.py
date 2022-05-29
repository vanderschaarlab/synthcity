# stdlib
from typing import Any

# third party
import numpy as np
import pandas as pd
import torch
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.nn.nets.resnet import ResidualNet
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseQuadraticAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import (
    AffineCouplingTransform,
    PiecewiseQuadraticCouplingTransform,
    PiecewiseRationalQuadraticCouplingTransform,
)
from nflows.transforms.lu import LULinear
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.svd import SVDLinear
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

# synthcity absolute
from synthcity.utils.constants import DEVICE


def create_alternating_binary_mask(features: int, even: bool = True) -> torch.Tensor:
    """
    Creates a binary mask of a given dimension which alternates its masking.
    :param features: Dimension of mask.
    :param even: If True, even values are assigned 1s, odd 0s. If False, vice versa.
    :return: Alternating binary mask of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    start = 0 if even else 1
    mask[start::2] += 1
    return mask


class NormalizingFlows(nn.Module):
    """Normalizing Flows are generative models which produce tractable distributions where both sampling and density evaluation can be efficient and exact.

    Args:
        n_iter: int
            Number of flow steps
        n_layers_hidden: int
            Number of transformation layers
        n_units_hidden: int
            Number of hidden units for each layer
        batch_size: int
            Size of batch used for training
        num_transform_blocks: int
            Number of blocks to use in coupling/autoregressive nets.
        dropout: float
            Dropout probability for coupling/autoregressive nets.
        batch_norm: bool
            Whether to use batch norm in coupling/autoregressive nets.
        num_bins: int
            Number of bins to use for piecewise transforms.
        tail_bound: float
            Box is on [-bound, bound]^2
        lr: float
            Learning rate for optimizer.
        apply_unconditional_transform: bool
            Whether to unconditionally transform \'identity\' features in the coupling layer.
        base_distribution: str
            Possible values: "standard_normal"
        linear_transform_type: str
            Type of linear transform to use. Possible values:
                - lu : A linear transform where we parameterize the LU decomposition of the weights.
                - permutation: Permutes using a random, but fixed, permutation.
                - svd: A linear module using the SVD decomposition for the weight matrix.
        base_transform_type: str
            Type of transform to use between linear layers. Possible values:
                - affine-coupling : An affine coupling layer that scales and shifts part of the variables.
                    Ref: L. Dinh et al., "Density estimation using Real NVP".
                - quadratic-coupling :
                    Ref: Müller et al., "Neural Importance Sampling".
                - rq-coupling : Rational Quadratic Coupling
                    Ref: Durkan et al, "Neural Spline Flows".
                - affine-autoregressive :Affine Autoregressive Transform
                    Ref: Durkan et al, "Neural Spline Flows".
                - quadratic-autoregressive : Quadratic Autoregressive Transform
                    Ref: Durkan et al, "Neural Spline Flows".
                - rq-autoregressive : Rational Quadratic Autoregressive Transform
                    Ref: Durkan et al, "Neural Spline Flows".
    """

    def __init__(
        self,
        n_iter: int = 1000,
        n_layers_hidden: int = 5,
        n_units_hidden: int = 10,
        batch_size: int = 100,
        num_transform_blocks: int = 2,
        dropout: float = 0.25,
        batch_norm: bool = False,
        num_bins: int = 8,
        tail_bound: float = 3,
        lr: float = 1e-3,
        apply_unconditional_transform: bool = True,
        base_distribution: str = "standard_normal",  # "standard_normal"
        linear_transform_type: str = "permutation",  # "lu", "permutation", "svd"
        base_transform_type: str = "rq-autoregressive",  # "affine-coupling", "quadratic-coupling", "rq-coupling", "affine-autoregressive", "quadratic-autoregressive", "rq-autoregressive"
        device: Any = DEVICE,
    ) -> None:
        super(NormalizingFlows, self).__init__()
        self.device = device
        self.n_iter = n_iter
        self.n_layers_hidden = n_layers_hidden
        self.n_units_hidden = n_units_hidden
        self.batch_size = batch_size
        self.num_transform_blocks = num_transform_blocks
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.apply_unconditional_transform = apply_unconditional_transform
        self.lr = lr

        self.base_distribution = base_distribution
        self.linear_transform_type = linear_transform_type
        self.base_transform_type = base_transform_type

    def dataloader(self, X: torch.Tensor) -> DataLoader:
        dataset = TensorDataset(X)
        return DataLoader(dataset, batch_size=self.batch_size, pin_memory=False)

    def generate(self, count: int) -> np.ndarray:
        return self(count).detach().cpu().numpy()

    def forward(self, count: int) -> torch.Tensor:
        self.flow.eval()
        with torch.no_grad():
            return self.flow.sample(count)

    def fit(self, X: pd.DataFrame) -> Any:
        # Load Dataset
        X = self._check_tensor(X).float().to(self.device)
        loader = self.dataloader(X)

        # Prepare flow
        features = X.shape[1]
        base_dist = self._get_base_distribution()(shape=[X.shape[1]]).to(self.device)

        transform = self._create_transform(features).to(self.device)

        self.flow = Flow(transform=transform, distribution=base_dist).to(self.device)

        # Prepare optimizer
        optimizer = optim.Adam(self.flow.parameters(), lr=self.lr)

        # Train

        for it in range(self.n_iter):
            for _, data in enumerate(loader):
                optimizer.zero_grad()
                loss = -self.flow.log_prob(inputs=data[0]).mean()
                loss.backward()
                optimizer.step()

        return self

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)

    def _get_base_distribution(self) -> Any:
        if self.base_distribution == "standard_normal":
            return StandardNormal
        else:
            raise ValueError(f"Unknown base distribution {self.base_distribution}")

    def _create_linear_transform(self, features: int) -> Any:
        if self.linear_transform_type == "permutation":
            return RandomPermutation(features=features)
        elif self.linear_transform_type == "lu":
            return CompositeTransform(
                [
                    RandomPermutation(features=features),
                    LULinear(features, identity_init=True),
                ]
            )
        elif self.linear_transform_type == "svd":
            return CompositeTransform(
                [
                    RandomPermutation(features=features),
                    SVDLinear(features, num_householder=10, identity_init=True),
                ]
            )
        else:
            raise ValueError(
                f"Unknown linear transform type {self.linear_transform_type}"
            )

    def _create_base_transform(self, layer_idx: int, features: int) -> Any:
        if self.base_transform_type == "affine-coupling":
            return AffineCouplingTransform(
                mask=create_alternating_binary_mask(
                    features, even=(layer_idx % 2 == 0)
                ),
                transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=self.n_units_hidden,
                    num_blocks=self.num_transform_blocks,
                    activation=F.relu,
                    dropout_probability=self.dropout,
                    use_batch_norm=self.batch_norm,
                ),
            )
        elif self.base_transform_type == "quadratic-coupling":
            return PiecewiseQuadraticCouplingTransform(
                mask=create_alternating_binary_mask(
                    features, even=(layer_idx % 2 == 0)
                ),
                transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=self.n_units_hidden,
                    num_blocks=self.num_transform_blocks,
                    activation=F.relu,
                    dropout_probability=self.dropout,
                    use_batch_norm=self.batch_norm,
                ),
                num_bins=self.num_bins,
                tails="linear",
                tail_bound=self.tail_bound,
                apply_unconditional_transform=self.apply_unconditional_transform,
            )
        elif self.base_transform_type == "rq-coupling":
            return PiecewiseRationalQuadraticCouplingTransform(
                mask=create_alternating_binary_mask(
                    features, even=(layer_idx % 2 == 0)
                ),
                transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=self.n_units_hidden,
                    num_blocks=self.num_transform_blocks,
                    activation=F.relu,
                    dropout_probability=self.dropout,
                    use_batch_norm=self.batch_norm,
                ),
                num_bins=self.num_bins,
                tails="linear",
                tail_bound=self.tail_bound,
                apply_unconditional_transform=self.apply_unconditional_transform,
            )
        elif self.base_transform_type == "affine-autoregressive":
            return MaskedAffineAutoregressiveTransform(
                features=features,
                hidden_features=self.n_units_hidden,
                num_blocks=self.num_transform_blocks,
                use_residual_blocks=True,
                random_mask=False,
                activation=F.relu,
                dropout_probability=self.dropout,
                use_batch_norm=self.batch_norm,
            )
        elif self.base_transform_type == "quadratic-autoregressive":
            return MaskedPiecewiseQuadraticAutoregressiveTransform(
                features=features,
                hidden_features=self.n_units_hidden,
                num_bins=self.num_bins,
                tails="linear",
                tail_bound=self.tail_bound,
                num_blocks=self.num_transform_blocks,
                use_residual_blocks=True,
                random_mask=False,
                activation=F.relu,
                dropout_probability=self.dropout,
                use_batch_norm=self.batch_norm,
            )
        elif self.base_transform_type == "rq-autoregressive":
            return MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=features,
                hidden_features=self.n_units_hidden,
                num_bins=self.num_bins,
                tails="linear",
                tail_bound=self.tail_bound,
                num_blocks=self.num_transform_blocks,
                use_residual_blocks=True,
                random_mask=False,
                activation=F.relu,
                dropout_probability=self.dropout,
                use_batch_norm=self.batch_norm,
            )
        else:
            raise ValueError(f"Unknown base transform {self.base_transform_type}")

    def _create_transform(self, features: int) -> Any:
        transform = CompositeTransform(
            [
                CompositeTransform(
                    [
                        self._create_linear_transform(features),
                        self._create_base_transform(layer_idx, features),
                    ]
                )
                for layer_idx in range(self.n_layers_hidden)
            ]
            + [self._create_linear_transform(features)]
        )
        return transform
