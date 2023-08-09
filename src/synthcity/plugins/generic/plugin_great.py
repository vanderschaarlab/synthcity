"""
Reference: "Language Models are Realistic Tabular Data Generators" Authors: Vadim Borisov and Kathrin Sessler and Tobias Leemann and Martin Pawelczyk and Gjergji Kasneci
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
from synthcity.plugins.core.distribution import (  # CategoricalDistribution,
    Distribution,
    IntegerDistribution,
)
from synthcity.plugins.core.models.tabular_great import TabularGReaT
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE


class GReaTPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.generic.plugin_great.GReaTPlugin
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
        # core plugin arguments
        device: Union[str, torch.device] = DEVICE,
        random_state: int = 0,
        sampling_patience: int = 500,
        workspace: Path = Path("workspace"),
        compress_dataset: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        .. inheritance-diagram:: synthcity.plugins.generic.plugin_great.GReaTPlugin
        :parts: 1

        Generation of Realistic Tabular data with pretrained Transformer-based language models (GReaT) implementation.
        Based on: https://openreview.net/forum?id=cEygmQNOeI

        Args:

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
        # self.num_trees = num_trees # TODO: check if this is needed for great

    @staticmethod
    def name() -> str:
        return "great"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="n_iter", low=50, high=500, step=50),
            # TODO: Add more parameters here like llm select from categorical, etc.
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "GReaTPlugin":
        """_summary_

        Args:
            X (DataLoader): _description_
            kwargs (Any): keyword arguments passed on to an SKLearn RandomForestClassifier

        Raises:
            NotImplementedError: _description_

        Returns:
            ARFPlugin: _description_
        """
        self.model = TabularGReaT(
            X.dataframe(),
            # self.num_trees,
            **kwargs,
        )
        if "cond" in kwargs and kwargs["cond"] is not None:
            raise NotImplementedError(
                "conditional generation is not currently available for the Generation of Realistic Tabular data with pretrained Transformer-based language models (GReaT) plugin."
            )
        self.model.fit(X.dataframe(), **kwargs)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        if "cond" in kwargs and kwargs["cond"] is not None:
            raise NotImplementedError(
                "conditional generation is not currently available for the Generation of Realistic Tabular data with pretrained Transformer-based language models (GReaT) plugin."
            )

        return self._safe_generate(self.model.generate, count, syn_schema)


plugin = GReaTPlugin
