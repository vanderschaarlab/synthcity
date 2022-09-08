# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# third party
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

# synthcity absolute
import synthcity.logger as log
from synthcity.metrics.representations.networks import build_network

# One-class loss functions
# ------------------------


def OneClassLoss(outputs: torch.Tensor, c: torch.Tensor) -> torch.Tensor:

    dist = torch.sum((outputs - c) ** 2, dim=1)
    loss = torch.mean(dist)

    return loss


def SoftBoundaryLoss(
    outputs: torch.Tensor, R: torch.Tensor, c: torch.Tensor, nu: torch.Tensor
) -> torch.Tensor:

    dist = torch.sum((outputs - c) ** 2, dim=1)
    scores = dist - R**2
    loss = R**2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))

    scores = dist
    loss = (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))

    return loss


# Base network
# ---------------------


class BaseNet(nn.Module):

    """Base class for all neural networks."""

    def __init__(self) -> None:

        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        """Forward pass logic

        :return: Network output
        """
        raise NotImplementedError


def get_radius(dist: torch.Tensor, nu: float) -> np.ndarray:

    """Optimally solve for radius R via the (1-nu)-quantile of distances."""

    return np.quantile(np.sqrt(dist.clone().data.float().cpu().numpy()), 1 - nu)


class OneClassLayer(BaseNet):
    def __init__(
        self,
        input_dim: int,
        rep_dim: int,
        center: torch.Tensor,
        num_layers: int = 4,
        num_hidden: int = 32,
        activation: str = "ReLU",
        dropout_prob: float = 0.2,
        dropout_active: bool = False,
        lr: float = 2e-3,
        epochs: int = 1000,
        warm_up_epochs: int = 20,
        train_prop: float = 1.0,
        weight_decay: float = 2e-3,
        Radius: float = 1,
        nu: float = 1e-2,
    ):

        super().__init__()

        self.rep_dim = rep_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.activation = activation
        self.dropout_prob = dropout_prob
        self.dropout_active = dropout_active
        self.train_prop = train_prop
        self.learningRate = lr
        self.epochs = epochs
        self.warm_up_epochs = warm_up_epochs
        self.weight_decay = weight_decay
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # Make this an option
        else:
            self.device = torch.device("cpu")
        # set up the network

        self.model = build_network(
            network_name="feedforward",
            params={
                "input_dim": input_dim,
                "rep_dim": rep_dim,
                "num_hidden": num_hidden,
                "activation": activation,
                "num_layers": num_layers,
                "dropout_prob": dropout_prob,
                "dropout_active": dropout_active,
                "LossFn": "SoftBoundary",
            },
        ).to(self.device)

        # create the loss function

        self.c = center.to(self.device)
        self.R = Radius
        self.nu = nu

        self.loss_fn = SoftBoundaryLoss

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.model(x)

        return x

    def fit(self, x_train: torch.Tensor) -> None:

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learningRate,
            weight_decay=self.weight_decay,
        )
        self.X = torch.tensor(x_train.reshape((-1, self.input_dim))).float()

        if self.train_prop != 1:
            x_train, x_val = (
                x_train[: int(self.train_prop * len(x_train))],
                x_train[int(self.train_prop * len(x_train)) :],
            )
            inputs_val = Variable(torch.from_numpy(x_val).to(self.device)).float()

        self.losses = []
        self.loss_vals = []

        for epoch in range(self.epochs):

            # Converting inputs and labels to Variable

            inputs = Variable(torch.from_numpy(x_train)).to(self.device).float()

            self.model.zero_grad()

            self.optimizer.zero_grad()

            # get output from the model, given the inputs
            outputs = self.model(inputs)

            # get loss for the predicted output

            self.loss = self.loss_fn(outputs=outputs, R=self.R, c=self.c, nu=self.nu)

            # self.c    = torch.mean(torch.tensor(outputs).float(), dim=0)

            # get gradients w.r.t to parameters
            self.loss.backward(retain_graph=True)
            self.losses.append(self.loss.detach().cpu().numpy())

            # update parameters
            self.optimizer.step()

            if self.train_prop != 1.0:
                with torch.no_grad():

                    # get output from the model, given the inputs
                    outputs = self.model(inputs_val)

                    # get loss for the predicted output

                    loss_val = self.loss_fn(
                        outputs=outputs, R=self.R, c=self.c, nu=self.nu
                    )

                    self.loss_vals.append(loss_val)

            if self.train_prop == 1:
                log.debug("epoch {}, loss {}".format(epoch, self.loss.item()))
            else:
                log.debug(
                    "epoch {:4}, train loss {:.4e}, val loss {:.4e}".format(
                        epoch, self.loss.item(), loss_val
                    )
                )
