# stdlib
from typing import Any, Optional

# third party
import pandas as pd
import torch
from pydantic import validate_arguments
from torch import nn

# synthcity absolute
from synthcity.metrics.weighted_metrics import WeightedMetrics
from synthcity.utils.constants import DEVICE

# synthcity relative
from .flows import NormalizingFlows
from .tabular_encoder import TabularEncoder


class TabularFlows(nn.Module):
    """
    Normalizing flow for tabular data.

    This class combines normalizing flow and tabular encoder to form a generative model for tabular data.

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
        # early stopping
        n_iter_print: int
            Number of iterations after which to print updates and check the validation loss.
        n_iter_min: int
            Minimum number of iterations to go through before starting early stopping
        patience: int
            Max number of iterations without any improvement before early stopping is trigged.
        patience_metric: WeightedMetrics
            Metric evaluator
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        X: pd.DataFrame,
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
        encoder_max_clusters: int = 20,
        encoder_whitelist: list = [],
        device: Any = DEVICE,
        # early stopping
        n_iter_min: int = 100,
        n_iter_print: int = 10,
        patience: int = 10,
        patience_metric: Optional[WeightedMetrics] = None,
    ) -> None:
        super(TabularFlows, self).__init__()
        self.columns = X.columns
        self.encoder = TabularEncoder(
            max_clusters=encoder_max_clusters, whitelist=encoder_whitelist
        ).fit(X)

        self.model = NormalizingFlows(
            n_iter=n_iter,
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            batch_size=batch_size,
            num_transform_blocks=num_transform_blocks,
            dropout=dropout,
            batch_norm=batch_norm,
            num_bins=num_bins,
            tail_bound=tail_bound,
            lr=lr,
            apply_unconditional_transform=apply_unconditional_transform,
            base_distribution=base_distribution,
            linear_transform_type=linear_transform_type,
            base_transform_type=base_transform_type,
            device=device,
            n_iter_min=n_iter_min,
            n_iter_print=n_iter_print,
            patience=patience,
            patience_metric=patience_metric,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def encode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.transform(X)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def decode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.inverse_transform(X)

    def get_encoder(self) -> TabularEncoder:
        return self.encoder

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        X: pd.DataFrame,
    ) -> Any:
        X_enc = self.encode(X)
        self.model.fit(
            X_enc,
        )
        return self

    def generate(self, count: int) -> pd.DataFrame:
        samples = self.model.generate(count)
        return self.decode(pd.DataFrame(samples))

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, count: int) -> torch.Tensor:
        return self.model.forward(count)
