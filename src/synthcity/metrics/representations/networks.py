# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# future
from __future__ import absolute_import, division, print_function

# stdlib
from typing import Any

# third party
import torch
from torch import nn

torch.manual_seed(1)

# Global variables

ACTIVATION_DICT = {
    "ReLU": torch.nn.ReLU(),
    "Hardtanh": torch.nn.Hardtanh(),
    "ReLU6": torch.nn.ReLU6(),
    "Sigmoid": torch.nn.Sigmoid(),
    "Tanh": torch.nn.Tanh(),
    "ELU": torch.nn.ELU(),
    "CELU": torch.nn.CELU(),
    "SELU": torch.nn.SELU(),
    "GLU": torch.nn.GLU(),
    "LeakyReLU": torch.nn.LeakyReLU(),
    "LogSigmoid": torch.nn.LogSigmoid(),
    "Softplus": torch.nn.Softplus(),
}


def build_network(network_name: str, params: dict) -> Any:

    if network_name == "feedforward":

        net = feedforward_network(params)

    return net


def feedforward_network(params: dict) -> Any:

    """Architecture for a Feedforward Neural Network

    Args:

        ::params::

        ::params["input_dim"]::
        ::params[""rep_dim""]::
        ::params["num_hidden"]::
        ::params["activation"]::
        ::params["num_layers"]::
        ::params["dropout_prob"]::
        ::params["dropout_active"]::
        ::params["LossFn"]::

    Returns:

        ::_architecture::

    """

    modules = []

    if params["dropout_active"]:

        modules.append(torch.nn.Dropout(p=params["dropout_prob"]))

    # Input layer

    modules.append(
        torch.nn.Linear(params["input_dim"], params["num_hidden"], bias=False)
    )
    modules.append(ACTIVATION_DICT[params["activation"]])

    # Intermediate layers

    for u in range(params["num_layers"] - 1):

        if params["dropout_active"]:

            modules.append(torch.nn.Dropout(p=params["dropout_prob"]))

        modules.append(
            torch.nn.Linear(params["num_hidden"], params["num_hidden"], bias=False)
        )
        modules.append(ACTIVATION_DICT[params["activation"]])

    # Output layer

    modules.append(torch.nn.Linear(params["num_hidden"], params["rep_dim"], bias=False))

    _architecture = nn.Sequential(*modules)

    return _architecture
