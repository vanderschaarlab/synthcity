"""
Reference: "SurvivalGAN: Generating time-to-event Data for Survival Analysis", B. Cebere*, A. Norcliffe*, F. Imrie, M. van der Schaar, AISTATS 2023
"""

# stdlib
from pathlib import Path
from typing import Any, List, Optional, Union

# third party
import numpy as np
import pandas as pd

# Necessary packages
from pydantic import validate_arguments

# synthcity absolute
import synthcity.logger as log
import synthcity.plugins as plugins
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.models.tabular_encoder import BinEncoder
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.survival_analysis._survival_pipeline import SurvivalPipeline
from synthcity.utils.constants import DEVICE
from synthcity.utils.samplers import ImbalancedDatasetSampler


class SurvivalGANPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.survival_analysis.plugin_survival_gan.SurvivalGANPlugin
        :parts: 1

    Survival Analysis Pipeline based on AdsGAN.

    Args:
        uncensoring_model: str
            The time-to-event model: "survival_function_regression".
        dataloader_sampling_strategy: str, default = imbalanced_time_censoring
            Training sampling strategy: none, imbalanced_censoring, imbalanced_time_censoring
        tte_strategy: str
            The time-to-event generation strategy: survival_function, uncensoring.
        censoring_strategy: str
            For the generated data, how to censor subjects: "random" or "covariate_dependent"
        device:
            torch device to use for training(cpu/cuda)
        kwargs: Any
            "adsgan" additional args, like n_iter = 100 etc.
        # Core Plugin arguments
        workspace: Path.
            Optional Path for caching intermediary results.
        compress_dataset: bool. Default = False.
            Drop redundant features before training the generator.
        sampling_patience: int.
            Max inference iterations to wait for the generated data to match the training schema.


    Example:
        >>> from lifelines.datasets import load_rossi
        >>> from synthcity.plugins import Plugins
        >>> from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader
        >>>
        >>> X = load_rossi()
        >>> data = SurvivalAnalysisDataLoader(
        >>>        X,
        >>>        target_column="arrest",
        >>>        time_to_event_column="week",
        >>> )
        >>>
        >>> plugin = Plugins().get("survival_gan")
        >>> plugin.fit(data)
        >>>
        >>> plugin.generate(count = 50)

    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        uncensoring_model: str = "survival_function_regression",
        dataloader_sampling_strategy: str = "imbalanced_time_censoring",  # none, imbalanced_censoring, imbalanced_time_censoring
        tte_strategy: str = "survival_function",
        censoring_strategy: str = "random",  # "covariate_dependent"
        device: Any = DEVICE,
        use_survival_conditional: bool = True,
        # core plugin arguments
        workspace: Path = Path("workspace"),
        random_state: int = 0,
        compress_dataset: bool = False,
        sampling_patience: int = 500,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            device=device,
            random_state=random_state,
            sampling_patience=sampling_patience,
            workspace=workspace,
            compress_dataset=compress_dataset,
        )

        if censoring_strategy not in [
            "random",
            "covariate_dependent",
        ]:
            raise ValueError(f"Invalid censoring strategy {censoring_strategy}")
        valid_sampling_strategies = [
            "none",
            "imbalanced_censoring",
            "imbalanced_time_censoring",
        ]
        if dataloader_sampling_strategy not in valid_sampling_strategies:
            raise ValueError(
                f"Invalid sampling strategy {dataloader_sampling_strategy}. Supported values: {valid_sampling_strategies}"
            )
        self.tte_strategy = tte_strategy
        self.dataloader_sampling_strategy = dataloader_sampling_strategy
        self.censoring_strategy = censoring_strategy
        self.uncensoring_model = uncensoring_model
        self.device = device
        self.use_survival_conditional = use_survival_conditional
        self.kwargs = kwargs
        self.random_state = random_state
        self.workspace = workspace
        self.compress_dataset = compress_dataset
        self.sampling_patience = sampling_patience

        log.info(
            f"""
            Training SurvivalGAN using
                dataloader_sampling_strategy = {self.dataloader_sampling_strategy};
                tte_strategy = {self.tte_strategy};
                uncensoring_model={self.uncensoring_model}
                censoring_strategy = {censoring_strategy}
                device={self.device}
            """
        )

    @staticmethod
    def name() -> str:
        return "survival_gan"

    @staticmethod
    def type() -> str:
        return "survival_analysis"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return plugins.Plugins().get_type("adsgan").hyperparameter_space()

    def _fit(
        self,
        X: DataLoader,
        *args: Any,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray, list]] = None,
        **kwargs: Any,
    ) -> "SurvivalGANPlugin":
        if X.type() != "survival_analysis":
            raise ValueError(f"Invalid data type = {X.type()}")

        sampler: Optional[ImbalancedDatasetSampler] = None
        sampling_labels: Optional[list] = None

        _, T, E = X.unpack()
        if self.dataloader_sampling_strategy == "imbalanced_censoring":
            log.info("Using imbalanced censoring sampling")
            sampling_labels = list(E.values)
        elif self.dataloader_sampling_strategy == "imbalanced_time_censoring":
            log.info("Using imbalanced time and censoring sampling")
            Tbins = BinEncoder().fit_transform(T.to_frame()).values.squeeze().tolist()
            sampling_labels = list(zip(E, Tbins))

        if sampling_labels is not None:
            sampler = ImbalancedDatasetSampler(sampling_labels)

        if cond is not None:
            cond = pd.DataFrame(cond)

        train_conditional = cond
        if self.use_survival_conditional and cond is not None:
            log.warning(
                "[SurvivalGAN] Using a user conditional will disable to built-in SurvivalGAN conditional"
            )
        if self.use_survival_conditional and cond is None:
            important_feats = X.important_features
            precond = pd.concat(
                [T.to_frame(), E.to_frame(), X[important_feats]], axis=1
            )
            train_conditional = BinEncoder().fit_transform(precond)

        self.model = SurvivalPipeline(
            "adsgan",
            strategy=self.tte_strategy,
            uncensoring_model=self.uncensoring_model,
            censoring_strategy=self.censoring_strategy,
            dataloader_sampler=sampler,
            device=self.device,
            random_state=self.random_state,
            workspace=self.workspace,
            compress_dataset=self.compress_dataset,
            sampling_patience=self.sampling_patience,
            **self.kwargs,
        )
        self.model.fit(X, cond=train_conditional, *args, **kwargs)

        return self

    def _generate(
        self,
        count: int,
        syn_schema: Schema,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray, list]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        return self.model._generate(
            count,
            syn_schema=syn_schema,
            cond=cond,
            **kwargs,
        )


plugin = SurvivalGANPlugin
