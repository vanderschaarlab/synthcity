# stdlib
from typing import Any, List

# third party
import pandas as pd

# Necessary packages
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.models import TabularVAE
from synthcity.plugins.models.time_to_event.tte_aft import WeibullAFTTimeToEvent
from synthcity.plugins.models.time_to_event.tte_coxph import CoxPHTimeToEvent
from synthcity.plugins.models.time_to_event.tte_date import DATETimeToEvent
from synthcity.plugins.models.time_to_event.tte_deephit import DeephitTimeToEvent
from synthcity.plugins.models.time_to_event.tte_rsf import (
    RandomSurvivalForestTimeToEvent,
)
from synthcity.plugins.models.time_to_event.tte_tenn import TENNTimeToEvent
from synthcity.plugins.models.time_to_event.tte_xgb import XGBTimeToEvent


class SurVAEPlugin(Plugin):
    """SurVAE plugin.

    Args:
        uncensoring_model: str
            The model class to use for uncensoring
        uncensoring_model_args: dict
            Parameters for the uncensoring model
        decoder_n_layers_hidden: int
            Number of hidden layers in the decoder
        decoder_n_units_hidden: int
            Number of hidden units in each layer of the decoder
        decoder_nonlin: string, default 'tanh'
            Nonlinearity to use in the decoder. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        decoder_dropout: float
            Dropout value. If 0, the dropout is not used.
        encoder_n_layers_hidden: int
            Number of hidden layers in the encoder
        encoder_n_units_hidden: int
            Number of hidden units in each layer of the encoder
        encoder_nonlin: string, default 'leaky_relu'
            Nonlinearity to use in the encoder. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        encoder_dropout: float
            Dropout value for the encoder. If 0, the dropout is not used.
        n_iter: int
            Maximum number of iterations in the encoder.
        lr: float
            learning rate for optimizer. step_size equivalent in the JAX version.
        weight_decay: float
            l2 (ridge) penalty for the weights.
        batch_size: int
            Batch size
        seed: int
            Seed used
        clipping_value: int, default 1
            Gradients clipping value
        encoder_max_clusters: int
            The max number of clusters to create for continuous columns when encoding

    Example:
        >>> from synthcity.plugins import Plugins
        >>> from lifelines.datasets import load_rossi
        >>> X = load_rossi()
        >>> plugin = Plugins().get("survae", event_column = "arrest", time_to_event_column="week")
        >>> plugin.fit(X)
        >>> plugin.generate()
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        uncensoring_model: str = "tenn",
        uncensoring_model_args: dict = {},
        event_column: str = "event",
        time_to_event_column: str = "duration",
        n_iter: int = 100,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        batch_size: int = 500,
        seed: int = 0,
        clipping_value: int = 1,
        loss_strategy: str = "robust_divergence",
        decoder_n_layers_hidden: int = 2,
        decoder_n_units_hidden: int = 100,
        decoder_nonlin: str = "tanh",
        decoder_dropout: float = 0,
        encoder_n_layers_hidden: int = 2,
        encoder_n_units_hidden: int = 100,
        encoder_nonlin: str = "leaky_relu",
        encoder_dropout: float = 0.1,
        data_encoder_max_clusters: int = 20,
        robust_divergence_beta: int = 2,  # used only for loss_strategy "robust_divergence"
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.event_column = event_column
        self.time_to_event_column = time_to_event_column

        self.uncensoring_model = self._get_uncensoring_model_template(
            uncensoring_model
        )(**uncensoring_model_args)

        self.decoder_n_layers_hidden = decoder_n_layers_hidden
        self.decoder_n_units_hidden = decoder_n_units_hidden
        self.decoder_nonlin = decoder_nonlin
        self.decoder_dropout = decoder_dropout
        self.encoder_n_layers_hidden = encoder_n_layers_hidden
        self.encoder_n_units_hidden = encoder_n_units_hidden
        self.encoder_nonlin = encoder_nonlin
        self.encoder_dropout = encoder_dropout
        self.n_iter = n_iter
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.seed = seed
        self.clipping_value = clipping_value
        self.data_encoder_max_clusters = data_encoder_max_clusters
        self.loss_strategy = loss_strategy

    @staticmethod
    def name() -> str:
        return "survae"

    @staticmethod
    def type() -> str:
        return "survival_analysis"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            CategoricalDistribution(
                name="uncensoring_model",
                choices=[
                    "tenn",
                    "date",
                    "weibull_aft",
                    "cox_ph",
                    "random_survival_forest",
                    "survival_xgboost",
                    "deephit",
                ],
            ),
            IntegerDistribution(name="n_iter", low=100, high=500, step=100),
            CategoricalDistribution(name="lr", choices=[1e-3, 2e-4, 1e-4]),
            IntegerDistribution(name="decoder_n_layers_hidden", low=1, high=5),
            CategoricalDistribution(name="weight_decay", choices=[1e-3, 1e-4]),
            CategoricalDistribution(name="batch_size", choices=[64, 128, 256, 512]),
            IntegerDistribution(
                name="decoder_n_units_hidden", low=50, high=500, step=50
            ),
            CategoricalDistribution(
                name="decoder_nonlin", choices=["relu", "leaky_relu", "tanh", "elu"]
            ),
            FloatDistribution(name="decoder_dropout", low=0, high=0.2),
            IntegerDistribution(name="encoder_n_layers_hidden", low=1, high=5),
            IntegerDistribution(
                name="encoder_n_units_hidden", low=50, high=500, step=50
            ),
            CategoricalDistribution(
                name="encoder_nonlin",
                choices=["relu", "leaky_relu", "tanh", "elu"],
            ),
            FloatDistribution(name="encoder_dropout", low=0, high=0.2),
        ]

    def _get_uncensoring_model_template(self, model_name: str) -> Any:
        defaults = {
            "tenn": TENNTimeToEvent,
            "date": DATETimeToEvent,
            "weibull_aft": WeibullAFTTimeToEvent,
            "cox_ph": CoxPHTimeToEvent,
            "random_survival_forest": RandomSurvivalForestTimeToEvent,
            "survival_xgboost": XGBTimeToEvent,
            "deephit": DeephitTimeToEvent,
        }

        if model_name in defaults:
            return defaults[model_name]

        raise RuntimeError(f"invalid model {model_name}")

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "SurVAEPlugin":
        if self.event_column not in X.columns:
            raise ValueError(
                f"Event column {self.event_column} not found in the dataframe"
            )

        if self.time_to_event_column not in X.columns:
            raise ValueError(
                f"Time to event column {self.time_to_event_column} not found in the dataframe"
            )

        Xcov = X.drop(columns=[self.event_column, self.time_to_event_column])
        T = X[self.time_to_event_column]
        E = X[self.event_column]

        # Uncensoring
        self.uncensoring_model.fit(Xcov, T, E)
        T_uncensored = pd.Series(self.uncensoring_model.predict(Xcov), index=Xcov.index)
        T_uncensored[E == 1] = T[E == 1]

        df_uncensored = Xcov
        df_uncensored[self.time_to_event_column] = T_uncensored

        features = df_uncensored.shape[1]

        # Synthetic data generator
        self.model = TabularVAE(
            df_uncensored,
            n_units_embedding=features,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            n_iter=self.n_iter,
            loss_strategy=self.loss_strategy,
            decoder_n_layers_hidden=self.decoder_n_layers_hidden,
            decoder_n_units_hidden=self.decoder_n_units_hidden,
            decoder_nonlin=self.decoder_nonlin,
            decoder_nonlin_out_discrete="softmax",
            decoder_nonlin_out_continuous="tanh",
            decoder_residual=True,
            decoder_batch_norm=False,
            decoder_dropout=0,
            encoder_n_units_hidden=self.encoder_n_units_hidden,
            encoder_n_layers_hidden=self.encoder_n_layers_hidden,
            encoder_nonlin=self.encoder_nonlin,
            encoder_batch_norm=False,
            encoder_dropout=self.encoder_dropout,
            clipping_value=self.clipping_value,
            encoder_max_clusters=self.data_encoder_max_clusters,
        )
        self.model.fit(df_uncensored)

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _generate(count: int) -> pd.DataFrame:
            generated = self.model.generate(count)
            generated[
                self.event_column
            ] = 1  # everybody is uncensored in the synthetic data
            return generated

        return self._safe_generate(_generate, count, syn_schema)


plugin = SurVAEPlugin
