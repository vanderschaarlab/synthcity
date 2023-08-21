"""
Reference: "Language Models are Realistic Tabular Data Generators" Authors: Vadim Borisov and Kathrin Sessler and Tobias Leemann and Martin Pawelczyk and Gjergji Kasneci
"""

# stdlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# third party
import pandas as pd
import torch

# Necessary packages
from pydantic import validate_arguments

# synthcity absolute
import synthcity.logger as log
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import Distribution, IntegerDistribution
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
        >>> plugin = Plugins().get("great")
        >>> plugin.fit(X)
        >>>
        >>> plugin.generate(50)

    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_iter: int = 100,
        llm: str = "distilgpt2",
        experiment_dir: str = "trainer_great",
        batch_size: int = 8,
        train_kwargs: Dict = {},
        # core plugin arguments
        logging_epoch: int = 100,
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
            n_iter: Optional[int]
                Number of iterations or epochs to train the model. Defaults to 100.
            llm: Optional[str]
                The language model to use. HuggingFace checkpoint of a pretrained large
                language model, used a basis of our model. Defaults to "distilgpt2".
            experiment_dir: Optional[str]
                The directory to save the model checkpoints and logs inside the workspace. Defaults to "trainer_great".
            batch_size: Optional[int]
                Batch size for training the model. Defaults to 8.
            train_kwargs: Optional[int]
                Additional hyperparameters added to the TrainingArguments used by the
                HuggingFaceLibrary, see here the full list of all possible values
                https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.
                Defaults to {}.
            conditional_col: [str] = "target"
                The column to condition the generation on. Defaults to "target".
            conditional_col_dist: Union[dict, List] = {}
                The distribution of the conditional column. Defaults to {}.
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
        self.n_iter = n_iter
        self.llm = llm
        self.experiment_dir = str((self.workspace / Path(experiment_dir)).resolve())
        self.batch_size = batch_size
        self.train_kwargs = train_kwargs

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
        ]

    def _fit(
        self,
        X: DataLoader,
        conditional_col: Optional[str] = None,
        resume_from_checkpoint: Union[str, bool] = False,
        *args: Any,
        **kwargs: Any,
    ) -> "GReaTPlugin":
        """
        Internal fit method for the GReaT plugin.

        Args:
            X (DataLoader): The data to fit the model on.

        Raises:
            NotImplementedError: _description_ # TODO: update this

        Returns:
            GReaTPlugin: The fitted plugin.
        """
        self.model = TabularGReaT(
            X.dataframe(),
            n_iter=self.n_iter,
            llm=self.llm,
            experiment_dir=self.experiment_dir,
            batch_size=self.batch_size,
            train_kwargs=self.train_kwargs,
        )
        if "cond" in kwargs and kwargs["cond"] is not None:
            raise NotImplementedError(
                "Conditional generation is not currently available for the Generation of Realistic Tabular data with pretrained Transformer-based language models (GReaT) plugin."
            )  # TODO: understand conditional col and update this
        self.model.fit(
            X.dataframe(),
            conditional_col=conditional_col,
            resume_from_checkpoint=resume_from_checkpoint,
        )
        log.debug(self.model)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        """
        Internal generate method for the GReaT plugin.

        Args:
            count (int): The number of rows to generate.
            syn_schema (Schema): The schema to generate data for.

        Raises:
            NotImplementedError: _description_ # TODO: update this

        Returns:
            pd.DataFrame: The generated data.
        """
        if "cond" in kwargs and kwargs["cond"] is not None:
            raise NotImplementedError(
                "conditional generation is not currently available for the Generation of Realistic Tabular data with pretrained Transformer-based language models (GReaT) plugin."
            )  # TODO: understand conditional col and update this

        return self._safe_generate(self.model.generate, count, syn_schema)


plugin = GReaTPlugin
