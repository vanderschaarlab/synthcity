"""
Reference: "Adversarial random forests for density estimation and generative modeling" Authors: David S. Watson, Kristin Blesch, Jan Kapar, and Marvin N. Wright
"""

# stdlib
from pathlib import Path
from typing import Any, List, Optional, Union

# third party
import pandas as pd
import torch

# Necessary packages
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import (
    Distribution,
    FloatDistribution,
    IntegerDistribution,
    LogDistribution,
)
from synthcity.plugins.core.models.tabular_aim import TabularAIM
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE


class AIMPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.privacy.plugin_adsgan.AdsGANPlugin
        :parts: 1

    Args:


    Example:
        >>> from sklearn.datasets import load_iris
        >>> from synthcity.plugins import Plugins
        >>>
        >>> X, y = load_iris(as_frame = True, return_X_y = True)
        >>> X["target"] = y
        >>>
        >>> plugin = Plugins().get("aim")
        >>> plugin.fit(X)
        >>>
        >>> plugin.generate(50)

    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        # AIM plugin arguments
        epsilon: float = 1.0,
        delta: float = 1e-9,
        max_model_size: int = 80,
        degree: int = 2,
        num_marginals: Optional[int] = None,
        max_cells: int = 1000,
        # core plugin arguments
        device: Union[str, torch.device] = DEVICE,
        random_state: int = 0,
        sampling_patience: int = 500,
        workspace: Path = Path("workspace"),
        compress_dataset: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        .. inheritance-diagram:: synthcity.plugins.generic.plugin_aim.AIMPlugin
        :parts: 1

        Adversarial Random Forest implementation.

        Args:
            degree: int = 2
                Degree of marginals to use. Defaults to 2.
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
        self.epsilon = epsilon
        self.delta = delta
        self.max_model_size = max_model_size
        self.degree = degree
        self.num_marginals = num_marginals
        self.max_cells = max_cells

    @staticmethod
    def name() -> str:
        return "aim"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            FloatDistribution(name="epsilon", low=0.5, high=3.0, step=0.5),
            LogDistribution(name="delta", low=1e-10, high=1e-5),
            IntegerDistribution(name="max_model_size", low=50, high=200, step=50),
            IntegerDistribution(name="degree", low=2, high=5, step=1),
            IntegerDistribution(name="num_marginals", low=0, high=5, step=1),
            IntegerDistribution(name="max_cells", low=5000, high=25000, step=5000),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "AIMPlugin":
        """
        Internal function to fit the model to the data.

        Args:
            X (DataLoader): The data to fit the model to.

        Raises:
            NotImplementedError: _description_

        Returns:
            AIMPlugin: _description_
        """

        self.model = TabularAIM(
            X.dataframe(),
            self.epsilon,
            self.delta,
            self.max_model_size,
            self.degree,
            self.num_marginals,
            self.max_cells,
            **kwargs,
        )
        if "cond" in kwargs and kwargs["cond"] is not None:
            raise NotImplementedError(
                "conditional generation is not currently available for the Adaptive and Iterative Mechanism for Differentially Private Synthetic Data(AIM) plugin."
            )
        self.model.fit(X.dataframe(), **kwargs)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        if "cond" in kwargs and kwargs["cond"] is not None:
            raise NotImplementedError(
                "conditional generation is not currently available for the Adaptive and Iterative Mechanism for Differentially Private Synthetic Data (AIM) plugin."
            )

        return self._safe_generate(self.model.generate, count, syn_schema)


plugin = AIMPlugin
