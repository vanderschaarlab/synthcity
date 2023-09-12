"""
Reference: "Adversarial random forests for density estimation and generative modeling" Authors: David S. Watson, Kristin Blesch, Jan Kapar, and Marvin N. Wright
"""

# stdlib
from pathlib import Path
from typing import Any, List, Union

# third party
import pandas as pd
import torch

# Necessary packages
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    IntegerDistribution,
)
from synthcity.plugins.core.models.tabular_arf import TabularARF
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE


class ARFPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.generic.plugin_arf.ARFPlugin
        :parts: 1

    Args:


    Example:
        >>> from sklearn.datasets import load_iris
        >>> from synthcity.plugins import Plugins
        >>>
        >>> X, y = load_iris(as_frame = True, return_X_y = True)
        >>> X["target"] = y
        >>>
        >>> plugin = Plugins().get("arf")
        >>> plugin.fit(X)
        >>>
        >>> plugin.generate(50)

    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        # n_iter: int = 1000,
        num_trees: int = 30,
        delta: int = 0,
        max_iters: int = 10,
        early_stop: bool = True,
        verbose: bool = True,
        min_node_size: int = 5,
        # core plugin arguments
        device: Union[str, torch.device] = DEVICE,
        random_state: int = 0,
        sampling_patience: int = 500,
        workspace: Path = Path("workspace"),
        compress_dataset: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        .. inheritance-diagram:: synthcity.plugins.generic.plugin_arf.ARFPlugin
        :parts: 1

        Adversarial Random Forest implementation.

        Args:
            n_iter: int = 1000
                Maximum number of iterations in the Generator. Defaults to 1000.
            learning_rate: float = 5e-3
                Generator learning rate, used by the Adam optimizer. Defaults to 5e-3.
            weight_decay: float = 1e-3
                Generator weight decay, used by the Adam optimizer. Defaults to 1e-3.
            batch_size: int = 32
                batch size. Defaults to 32.
            patience: int = 50
                Max number of iterations without any improvement before early stopping is triggered. Defaults to 50.
            logging_epoch: int = 100
                The number of epochs after which information is sent to the debugging level of the logger. Defaults to 100.
            device: Union[str, torch.device] = synthcity.utils.constants.DEVICE
                The device that the model is run on. Defaults to "cuda" if cuda is available else "cpu".
            random_state: int = 0
                random_state used. Defaults to 0.
            sampling_patience: int = 500
                Max inference iterations to wait for the generated data to match the training schema. Defaults to 500.
            workspace: Path
                Path for caching intermediary results. Defaults to Path("workspace").
            compress_dataset: bool. Default = False
                Drop redundant features before training the generator. Defaults to False.
            dataloader_sampler: Any = None
                Optional sampler for the dataloader. Defaults to None.
        """
        super().__init__(
            device=device,
            random_state=random_state,
            sampling_patience=sampling_patience,
            workspace=workspace,
            compress_dataset=compress_dataset,
            **kwargs,
        )
        self.num_trees = num_trees
        self.delta = delta
        self.max_iters = max_iters
        self.early_stop = early_stop
        self.verbose = verbose
        self.min_node_size = min_node_size

    @staticmethod
    def name() -> str:
        return "arf"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="num_trees", low=10, high=100, step=10),
            IntegerDistribution(
                name="delta", low=0, high=50, step=2
            ),  # TODO: check if this is the right range
            IntegerDistribution(name="max_iters", low=1, high=5, step=1),
            CategoricalDistribution(name="early_stop", choices=[True, False]),
            IntegerDistribution(name="min_node_size", low=2, high=20, step=2),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "ARFPlugin":
        """_summary_

        Args:
            X (DataLoader): _description_
            kwargs (Any): keyword arguments passed on to an SKLearn RandomForestClassifier

        Raises:
            NotImplementedError: _description_

        Returns:
            ARFPlugin: _description_
        """
        self.model = TabularARF(
            X.dataframe(),
            self.num_trees,
            self.delta,
            self.max_iters,
            self.early_stop,
            self.verbose,
            self.min_node_size,
            **kwargs,
        )
        if "cond" in kwargs and kwargs["cond"] is not None:
            raise NotImplementedError(
                "conditional generation is not currently available for the Adversarial Random Forest (ARF) plugin."
            )
        self.model.fit(X.dataframe(), **kwargs)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        if "cond" in kwargs and kwargs["cond"] is not None:
            raise NotImplementedError(
                "conditional generation is not currently available for the Adversarial Random Forest (ARF) plugin."
            )

        return self._safe_generate(self.model.generate, count, syn_schema)


plugin = ARFPlugin
