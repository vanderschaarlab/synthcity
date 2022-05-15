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
        dataloader_sampling_strategy: str = "imbalanced_full",  # none, imbalanced_censoring, imbalanced_time_censoring, imbalanced_full, imbalanced_cov_censoring
        tte_strategy: str = "survival_function",
        **kwargs: Any,
    ) -> None:
        super().__init__()

        valid_sampling_strategies = [
            "none",
            "imbalanced_censoring",
            "imbalanced_time_censoring",
            "imbalanced_cov_censoring",
            "imbalanced_full",
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

        self.kwargs = kwargs

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
        elif self.dataloader_sampling_strategy == "imbalanced_cov_censoring":
            log.info("Using imbalanced covariates and censoring sampling")
            bins = BinEncoder().fit_transform(X)
            bins = bins.drop(columns=[self.time_to_event_column])
            bins_arr = []
            for col in bins.columns:
                bins_arr.append(bins[col].values.tolist())
            sampling_labels = list(zip(*bins_arr))
        elif self.dataloader_sampling_strategy == "imbalanced_full":
            log.info("Using full imbalanced sampling")
            bins = BinEncoder().fit_transform(X)
            bins_arr = []
            for col in bins.columns:
                bins_arr.append(bins[col].values.tolist())
            sampling_labels = list(zip(*bins_arr))

        if sampling_labels is not None:
            sampler = ImbalancedDatasetSampler(sampling_labels)

        self.model = SurvivalPipeline(
            "adsgan",
            strategy=self.tte_strategy,
            target_column=self.target_column,
            time_to_event_column=self.time_to_event_column,
            time_horizons=self.time_horizons,
            uncensoring_model=self.uncensoring_model,
            dataloader_sampler=sampler,
            **self.kwargs,
        )
        self.model.fit(X, *args, **kwargs)

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self.model._generate(count, syn_schema=syn_schema, **kwargs)


plugin = SurvivalGANPlugin
