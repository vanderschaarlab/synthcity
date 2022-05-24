# stdlib
from typing import Any, List, Optional

# third party
import pandas as pd

# Necessary packages
from pydantic import validate_arguments

# synthcity absolute
import synthcity.logger as log
import synthcity.plugins as plugins
from synthcity.plugins._survival_pipeline import SurvivalPipeline
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.models import BinEncoder
from synthcity.utils.constants import DEVICE
from synthcity.utils.samplers import ImbalancedDatasetSampler


class SurvivalGANPlugin(Plugin):
    """Survival GAN plugin.

    Example:
        >>> from synthcity.plugins import Plugins
        >>> X = load_rossi()
        >>> plugin = Plugins().get("survival_gan", target_column = "arrest", time_to_event_column="week")
        >>> plugin.fit(X)
        >>> plugin.generate()
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        target_column: str = "event",
        time_to_event_column: str = "duration",
        time_horizons: Optional[List] = None,
        uncensoring_model: str = "survival_function_regression",
        dataloader_sampling_strategy: str = "imbalanced_time_censoring",  # none, imbalanced_censoring, imbalanced_time_censoring
        tte_strategy: str = "survival_function",
        device: Any = DEVICE,
        identifiability_penalty: bool = False,
        use_conditional: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        valid_sampling_strategies = [
            "none",
            "imbalanced_censoring",
            "imbalanced_time_censoring",
        ]
        if dataloader_sampling_strategy not in valid_sampling_strategies:
            raise ValueError(
                f"Invalid sampling strategy {dataloader_sampling_strategy}. Supported values: {valid_sampling_strategies}"
            )
        self.target_column = target_column
        self.time_to_event_column = time_to_event_column
        self.time_horizons = time_horizons
        self.tte_strategy = tte_strategy
        self.dataloader_sampling_strategy = dataloader_sampling_strategy
        self.uncensoring_model = uncensoring_model
        self.device = device
        self.use_conditional = use_conditional

        if identifiability_penalty:
            self.generator_extra_penalties = ["identifiability_penalty"]
        else:
            self.generator_extra_penalties = []

        self.kwargs = kwargs

        log.info(
            f"""
            Training SurvivalGAN using
                dataloader_sampling_strategy = {self.dataloader_sampling_strategy};
                tte_strategy = {self.tte_strategy};
                uncensoring_model={self.uncensoring_model}
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

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "SurvivalGANPlugin":
        sampler: Optional[ImbalancedDatasetSampler] = None
        sampling_labels: Optional[list] = None

        E = X[self.target_column]
        T = X[self.time_to_event_column]
        if self.dataloader_sampling_strategy == "imbalanced_censoring":
            log.info("Using imbalanced censoring sampling")
            sampling_labels = list(E.values)
        elif self.dataloader_sampling_strategy == "imbalanced_time_censoring":
            log.info("Using imbalanced time and censoring sampling")
            Tbins = BinEncoder().fit_transform(T.to_frame()).values.squeeze().tolist()
            sampling_labels = list(zip(E, Tbins))

        if sampling_labels is not None:
            sampler = ImbalancedDatasetSampler(sampling_labels)

        if self.use_conditional:
            self.conditional = BinEncoder().fit_transform(X)
            n_units_conditional = self.conditional.shape[1]
        else:
            self.conditional = None
            n_units_conditional = 0
        self.model = SurvivalPipeline(
            "adsgan",
            strategy=self.tte_strategy,
            target_column=self.target_column,
            time_to_event_column=self.time_to_event_column,
            time_horizons=self.time_horizons,
            uncensoring_model=self.uncensoring_model,
            dataloader_sampler=sampler,
            generator_extra_penalties=self.generator_extra_penalties,
            n_units_conditional=n_units_conditional,
            device=self.device,
            **self.kwargs,
        )
        self.model.fit(X, cond=self.conditional, *args, **kwargs)

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        cond = (
            self.conditional.sample(count, replace=True)
            if self.use_conditional
            else None
        )

        return self.model._generate(
            count,
            syn_schema=syn_schema,
            cond=cond,
            **kwargs,
        )


plugin = SurvivalGANPlugin
