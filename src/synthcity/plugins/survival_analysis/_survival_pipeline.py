# stdlib
from pathlib import Path
from typing import Any, List, Optional, Union

# third party
import numpy as np
import pandas as pd

# Necessary packages
from pydantic import validate_arguments
from xgboost import XGBClassifier

# synthcity absolute
import synthcity.logger as log
import synthcity.plugins as plugins
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.models.time_to_event import TimeToEventPlugin
from synthcity.plugins.core.models.time_to_event import (
    get_model_template as get_tte_model_template,
)
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE


class SurvivalPipeline(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.survival_analysis._survival_pipeline.SurvivalPipeline
        :parts: 1


    Survival uncensoring plugin pipeline.

    Args:
        method: str
            Baseline generator to use, e.g.: adsgan, ctgan etc.
        strategy: str
            The time-to-event generation strategy: survival_function, uncensoring.
        uncensoring_model: str
            The time-to-event model: "survival_function_regression".
        censoring_strategy: str
            For the generated data, how to censor subjects: "random" or "covariate_dependent"
        kwargs: Any
            The "method" additional args, like n_iter = 100 etc.
        # Core Plugin arguments
        workspace: Path.
            Optional Path for caching intermediary results.
        compress_dataset: bool. Default = False.
            Drop redundant features before training the generator.
        sampling_patience: int.
            Max inference iterations to wait for the generated data to match the training schema.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        method: str = "adsgan",
        strategy: str = "survival_function",  # uncensoring, survival_function
        uncensoring_model: str = "survival_function_regression",
        censoring_strategy: str = "random",  # "covariate_dependent"
        device: Any = DEVICE,
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

        self.device = device
        self.strategy = strategy
        self.censoring_strategy = censoring_strategy

        self.uncensoring_model: Optional[TimeToEventPlugin] = None
        if uncensoring_model != "none":
            self.uncensoring_model = get_tte_model_template(uncensoring_model)(
                device=device
            )

        self.generator = plugins.Plugins().get(
            method,
            device=device,
            workspace=workspace,
            random_state=random_state,
            compress_dataset=compress_dataset,
            sampling_patience=sampling_patience,
            **kwargs,
        )

    @staticmethod
    def name() -> str:
        return "survival_pipeline"

    @staticmethod
    def type() -> str:
        return "survival_analysis"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return []

    def _fit(
        self,
        X: DataLoader,
        *args: Any,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray, list]] = None,
        **kwargs: Any,
    ) -> "SurvivalPipeline":
        if X.type() != "survival_analysis":
            raise ValueError(f"Invalid data type {X.type()}")

        Xcov, T, E = X.unpack()

        self.last_te = T[E == 1].max()
        self.censoring_ratio = (E == 0).sum() / len(E)
        self.features = X.columns

        data_info = X.info()
        self.time_to_event_column = data_info["time_to_event_column"]
        self.target_column = data_info["target_column"]
        self.train_conditional = cond

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

            self.generator.fit(df_train, cond=cond, **kwargs)
        elif self.strategy == "survival_function":
            # Synthetic data generator
            self.generator.fit(X.dataframe(), cond=cond, **kwargs)
        else:
            raise ValueError(f"unsupported strategy {self.strategy}")

        xgb_params = {
            "n_jobs": 2,
            "verbosity": 0,
            "depth": 3,
            "random_state": 0,
        }
        self.censoring_predictor = XGBClassifier(**xgb_params).fit(Xcov, E)

        return self

    def _generate(
        self,
        count: int,
        syn_schema: Schema,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray, list]] = None,
        **kwargs: Any,
    ) -> DataLoader:

        gen_conditional: Optional[Union[pd.DataFrame, pd.Series]] = None
        if cond is not None:
            gen_conditional = pd.DataFrame(cond)
        elif self.train_conditional is not None:
            gen_conditional = pd.DataFrame(self.train_conditional)
            while len(gen_conditional) < count:
                gen_conditional = pd.concat(
                    [gen_conditional, gen_conditional], ignore_index=True
                )
            gen_conditional = gen_conditional.head(count)

        def _generate(count: int) -> pd.DataFrame:

            generated = self.generator.generate(
                count, cond=gen_conditional, **kwargs
            ).dataframe()
            if self.censoring_strategy == "covariate_dependent":
                generated[self.target_column] = self.censoring_predictor.predict(
                    generated.drop(
                        columns=[self.target_column, self.time_to_event_column]
                    )
                )

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
