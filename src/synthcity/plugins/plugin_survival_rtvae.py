# stdlib
from typing import Any, List, Optional

# third party
import pandas as pd

# Necessary packages
from pydantic import validate_arguments

# synthcity absolute
import synthcity.plugins as plugins
from synthcity.plugins._survival_pipeline import SurvivalPipeline
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.models.time_to_event.samplers import ImbalancedDatasetSampler


class SurvivalRTVAEPlugin(Plugin):
    """Survival RTVAE plugin.

    Example:
        >>> from synthcity.plugins import Plugins
        >>> X = load_rossi()
        >>> plugin = Plugins().get("survival_rtvae", target_column = "arrest", time_to_event_column="week")
        >>> plugin.fit(X)
        >>> plugin.generate()
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        strategy: str = "survival_function",
        target_column: str = "event",
        time_to_event_column: str = "duration",
        time_horizons: Optional[List] = None,
        uncensoring_seeds: List[str] = [
            "weibull_aft",
            "cox_ph",
            "random_survival_forest",
            "survival_xgboost",
            "deephit",
            "tenn",
            "date",
        ],
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.target_column = target_column
        self.time_to_event_column = time_to_event_column
        self.time_horizons = time_horizons
        self.strategy = strategy
        self.uncensoring_seeds = uncensoring_seeds
        self.kwargs = kwargs

    @staticmethod
    def name() -> str:
        return "survival_rtvae"

    @staticmethod
    def type() -> str:
        return "survival_analysis"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return plugins.Plugins().get_type("rtvae").hyperparameter_space()

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "SurvivalRTVAEPlugin":
        E = X[self.target_column]

        self.model = SurvivalPipeline(
            "rtvae",
            strategy=self.strategy,
            target_column=self.target_column,
            time_to_event_column=self.time_to_event_column,
            time_horizons=self.time_horizons,
            uncensoring_seeds=self.uncensoring_seeds,
            dataloader_sampler=ImbalancedDatasetSampler(E.values.tolist()),
            **self.kwargs,
        )
        self.model.fit(X, *args, **kwargs)

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self.model._generate(count, syn_schema=syn_schema, **kwargs)


plugin = SurvivalRTVAEPlugin
