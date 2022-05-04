# stdlib
from typing import Any, List, Optional

# third party
import pandas as pd

# Necessary packages
from pydantic import validate_arguments

# synthcity absolute
import synthcity.plugins as plugins
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.models.time_to_event import select_uncensoring_model


class SurvivalUncensoringPipeline(Plugin):
    """Survival uncensoring plugin pipeline."""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        method: str,
        target_column: str = "event",
        time_to_event_column: str = "duration",
        time_horizons: Optional[List] = None,
        seeds: List[str] = [
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
        self.seeds = seeds

        self.model = plugins.Plugins().get(method, **kwargs)

    @staticmethod
    def name() -> str:
        return "survival_uncensoring_pipeline"

    @staticmethod
    def type() -> str:
        return "survival_analysis"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return []

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "SurvivalUncensoringPipeline":
        if self.target_column not in X.columns:
            raise ValueError(
                f"Event column {self.target_column} not found in the dataframe"
            )

        if self.time_to_event_column not in X.columns:
            raise ValueError(
                f"Time to event column {self.time_to_event_column} not found in the dataframe"
            )

        Xcov = X.drop(columns=[self.target_column, self.time_to_event_column])
        T = X[self.time_to_event_column]
        E = X[self.target_column]

        # Uncensoring
        self.uncensoring_model = select_uncensoring_model(Xcov, T, E, seeds=self.seeds)

        self.uncensoring_model.fit(Xcov, T, E)
        T_uncensored = pd.Series(self.uncensoring_model.predict(Xcov), index=Xcov.index)
        T_uncensored[E == 1] = T[E == 1]

        df_uncensored = Xcov
        df_uncensored[self.time_to_event_column] = T_uncensored

        # Synthetic data generator
        self.model.fit(df_uncensored)

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _generate(count: int) -> pd.DataFrame:
            generated = self.model.generate(count)
            generated[
                self.target_column
            ] = 1  # everybody is uncensored in the synthetic data
            return generated

        return self._safe_generate(_generate, count, syn_schema)


plugin = SurvivalUncensoringPipeline
