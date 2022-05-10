# stdlib
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd

# Necessary packages
from pydantic import validate_arguments
from xgboost import XGBRegressor

# synthcity absolute
import synthcity.plugins as plugins
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.models.survival_analysis.loader import get_model_template
from synthcity.plugins.models.time_to_event import select_uncensoring_model


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
        uncensoring_seeds: List[str] = [
            "weibull_aft",
            "cox_ph",
            "random_survival_forest",
            "survival_xgboost",
            "deephit",
            "tenn",
            "date",
        ],
        survival_model: str = "cox_ph",
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.strategy = strategy

        self.target_column = target_column
        self.time_to_event_column = time_to_event_column

        self.uncensoring_seeds = uncensoring_seeds
        self.survival_model = survival_model

        self.generator = plugins.Plugins().get(method, **kwargs)

    @staticmethod
    def name() -> str:
        return "survival_pipeline"

    @staticmethod
    def type() -> str:
        return "survival_analysis"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "SurvivalPipeline":
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

        self.last_te = T[E == 1].max()
        self.censoring_ratio = (E == 0).sum() / len(E)

        if self.strategy == "uncensoring":
            # Uncensoring
            self.uncensoring_model = select_uncensoring_model(
                Xcov, T, E, seeds=self.uncensoring_seeds
            )
            self.uncensoring_model.fit(Xcov, T, E)
            T_uncensored = pd.Series(
                self.uncensoring_model.predict(Xcov), index=Xcov.index
            )
            T_uncensored[E == 1] = T[E == 1]

            df_train = Xcov.copy()
            df_train[self.time_to_event_column] = T_uncensored

            self.generator.fit(df_train)
        elif self.strategy == "survival_function":
            # Survival function driven
            self.surv_model = get_model_template(self.survival_model)().fit(Xcov, T, E)
            self.time_horizons = np.linspace(T.min(), T.max(), 10, dtype=int).tolist()
            surv_fn = self.surv_model.predict(Xcov, time_horizons=self.time_horizons)
            surv_fn[self.target_column] = E

            # Censoring proba
            xgb_params = {
                "n_jobs": 1,
                "verbosity": 0,
                "depth": 3,
                "random_state": 0,
            }

            self.tte_regressor = XGBRegressor(**xgb_params).fit(surv_fn, T)

            # Synthetic data generator
            self.generator.fit(X)
        else:
            raise ValueError(f"unsupported strategy {self.strategy}")

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _generate(count: int) -> pd.DataFrame:

            if self.strategy == "uncensoring":
                generated = self.generator.generate(count)
                generated[self.target_column] = 1
            elif self.strategy == "survival_function":

                def _prepare(label: int, count: int) -> pd.DataFrame:
                    constraints = Constraints(
                        rules=[
                            (self.target_column, "in", [label]),
                        ]
                    )
                    generated = self.generator.generate(count, constraints=constraints)
                    if len(generated) == 0:
                        return generated

                    generated = generated.drop(
                        columns=[self.time_to_event_column]
                    )  # remove the generated column

                    surv_f = self.surv_model.predict(
                        generated.drop(columns=[self.target_column]),
                        time_horizons=self.time_horizons,
                    )
                    surv_f[self.target_column] = label

                    generated[self.time_to_event_column] = self.tte_regressor.predict(
                        surv_f
                    )

                    return generated

                # Censored
                cens_count = int(self.censoring_ratio * count)
                cens_generated = _prepare(0, cens_count)

                # Non-censored
                noncens_count = max(int(0.5 * count), count - cens_count)
                noncens_generated = _prepare(1, noncens_count)

                # Output
                generated = pd.concat(
                    [cens_generated, noncens_generated], ignore_index=True
                )
            else:
                raise ValueError(f"unsupported strategy {self.strategy}")

            return generated

        return self._safe_generate(_generate, count, syn_schema)


plugin = SurvivalPipeline
