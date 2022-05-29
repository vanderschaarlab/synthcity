# stdlib
from typing import Any, List, Optional, Tuple

# third party
import pandas as pd

# Necessary packages
from pydantic import validate_arguments

# synthcity absolute
import synthcity.logger as log
import synthcity.plugins as plugins
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.models.time_to_event import TimeToEventPlugin
from synthcity.plugins.core.models.time_to_event import (
    get_model_template as get_tte_model_template,
)
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE


class SurvivalPipeline(Plugin):
    """Survival uncensoring plugin pipeline."""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        method: str,
        strategy: str = "survival_function",  # uncensoring, survival_function
        target_column: str = "event",
        time_to_event_column: str = "duration",
        time_horizons: Optional[List] = None,
        uncensoring_model: str = "survival_function_regression",
        device: Any = DEVICE,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.device = device
        self.strategy = strategy

        self.target_column = target_column
        self.time_to_event_column = time_to_event_column

        self.uncensoring_model: Optional[TimeToEventPlugin] = None
        if uncensoring_model != "none":
            self.uncensoring_model = get_tte_model_template(uncensoring_model)(
                device=device
            )

        self.generator = plugins.Plugins().get(method, device=device, **kwargs)

    @staticmethod
    def name() -> str:
        return "survival_pipeline"

    @staticmethod
    def type() -> str:
        return "survival_analysis"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return []

    def _preprocess(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        X = df.drop(columns=[self.target_column, self.time_to_event_column])
        T = df[self.time_to_event_column]
        E = df[self.target_column]

        return X, T, E

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "SurvivalPipeline":
        if self.target_column not in X.columns:
            raise ValueError(
                f"Event column {self.target_column} not found in the dataframe"
            )

        if self.time_to_event_column not in X.columns:
            raise ValueError(
                f"Time to event column {self.time_to_event_column} not found in the dataframe"
            )

        Xcov, T, E = self._preprocess(X)

        self.last_te = T[E == 1].max()
        self.censoring_ratio = (E == 0).sum() / len(E)
        self.features = X.columns

        if self.uncensoring_model is not None:
            log.info("Train the uncensoring model")
            self.uncensoring_model.fit(Xcov, T, E)

        log.info("Train the synthetic generator")
        if self.strategy == "uncensoring":
            if self.uncensoring_model is None:
                raise RuntimeError("Uncensoring strategies needs a TTE model")
            # Uncensoring
            T_uncensored = pd.Series(
                self.uncensoring_model.predict(Xcov), index=Xcov.index
            )
            T_uncensored[E == 1] = T[E == 1]

            df_train = Xcov.copy()
            df_train[self.time_to_event_column] = T_uncensored

            self.generator.fit(df_train, **kwargs)
        elif self.strategy == "survival_function":
            # Synthetic data generator
            self.generator.fit(X, **kwargs)
        else:
            raise ValueError(f"unsupported strategy {self.strategy}")

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _generate(count: int) -> pd.DataFrame:

            generated = self.generator.generate(count, **kwargs)
            if self.strategy == "uncensoring":
                generated[self.target_column] = 1
            elif self.strategy == "survival_function":
                if self.uncensoring_model is not None:
                    generated = generated.drop(
                        columns=[self.time_to_event_column]
                    )  # remove the generated column

                    generated[
                        self.time_to_event_column
                    ] = self.uncensoring_model.predict_any(
                        generated.drop(columns=[self.target_column]),
                        generated[self.target_column],
                    )
            else:
                raise ValueError(f"unsupported strategy {self.strategy}")

            return generated

        return self._safe_generate(_generate, count, syn_schema)


plugin = SurvivalPipeline
