"""
Fourier-Flows method based on " Generative Time-series Modeling with Fourier Flows", Ahmed Alaa, Alex Chan, and Mihaela van der Schaar.
"""
# stdlib
from typing import Any, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
from fflows import FourierFlow

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    IntegerDistribution,
)
from synthcity.plugins.core.models.tabular_encoder import (
    TabularEncoder,
    TimeSeriesTabularEncoder,
)
from synthcity.plugins.core.models.ts_rnn import TimeSeriesRNN
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.generic import GenericPlugins
from synthcity.utils.constants import DEVICE


class FourierFlowsPlugin(Plugin):
    """Synthetic time series generation using FourierFlows.

    Args:

    Example:
        >>> from synthcity.plugins import Plugins
        >>> from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
        >>> from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
        >>>
        >>> plugin = Plugins().get("fflows")
        >>> static, temporal, outcome = GoogleStocksDataloader(as_numpy=True).load()
        >>> loader = TimeSeriesDataLoader(
        >>>             temporal_data=temporal_data,
        >>>             static_data=static_data,
        >>>             outcome=outcome,
        >>> )
        >>> plugin.fit(loader)
        >>> plugin.generate()
    """

    def __init__(
        self,
        n_iter: int = 500,
        batch_size: int = 128,
        lr: float = 1e-3,
        n_iter_print: int = 100,
        n_units_hidden: int = 100,
        n_flows: int = 10,
        FFT: bool = True,
        flip: bool = True,
        normalize: bool = False,
        static_model: str = "ctgan",
        device: Any = DEVICE,
        encoder_max_clusters: int = 10,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.train_args = {
            "epochs": n_iter,
            "batch_size": batch_size,
            "learning_rate": lr,
            "display_step": n_iter_print,
        }
        self.temporal_model = FourierFlow(
            hidden=n_units_hidden,
            n_flows=n_flows,
            FFT=FFT,
            flip=flip,
            normalize=normalize,
        ).to(device)
        self.static_model: Optional[Plugin] = None
        self.static_model_name = static_model

        self.device = device
        self.temporal_encoder = TimeSeriesTabularEncoder(
            max_clusters=encoder_max_clusters
        )
        self.outcome_encoder = TabularEncoder(max_clusters=encoder_max_clusters)

    @staticmethod
    def name() -> str:
        return "fflows"

    @staticmethod
    def type() -> str:
        return "time_series"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="n_iter", low=100, high=1000, step=100),
            CategoricalDistribution(name="lr", choices=[1e-3, 2e-4, 1e-4]),
            CategoricalDistribution(name="batch_size", choices=[100, 200, 500]),
            IntegerDistribution(name="n_units_hidden", low=50, high=150, step=50),
            IntegerDistribution(name="n_flows", low=5, high=100),
            CategoricalDistribution(name="FFT", choices=[True, False]),
            CategoricalDistribution(name="flip", choices=[True, False]),
            CategoricalDistribution(name="normalize", choices=[True, False]),
            CategoricalDistribution(
                name="static_model", choices=["ctgan", "adsgan", "privbayes"]
            ),
            IntegerDistribution(name="encoder_max_clusters", low=2, high=20),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "FourierFlowsPlugin":
        assert X.type() == "time_series"

        # Train static generator
        static, temporal, outcome = X.unpack()

        self.temporal_encoder.fit(static, temporal)
        static_enc, temporal_enc = self.temporal_encoder.transform(static, temporal)

        if np.prod(static.shape) != 0:
            self.static_model = (
                GenericPlugins()
                .get(self.static_model_name, device=self.device)
                .fit(static_enc)
            )

        # Train temporal generator
        self.temporal_model.fit(temporal_enc, **self.train_args)

        # Train outcome generator
        self.outcome_encoder.fit(outcome)
        outcome_enc = self.outcome_encoder.transform(outcome)

        self.outcome_model = TimeSeriesRNN(
            task_type="regression",
            n_static_units_in=static_enc.shape[-1],
            n_temporal_units_in=temporal_enc[0].shape[-1],
            output_shape=outcome_enc.shape[1:],
            n_iter=self.train_args["epochs"],
            batch_size=self.train_args["batch_size"],
            lr=self.train_args["learning_rate"],
            device=self.device,
            nonlin_out=self.outcome_encoder.activation_layout(
                discrete_activation="softmax",
                continuous_activation="tanh",
            ),
        )
        self.outcome_model.fit(
            np.asarray(static_enc), np.asarray(temporal_enc), np.asarray(outcome_enc)
        )

        self.temporal_encoded_columns = temporal_enc[0].columns
        self.static_encoded_columns = static_enc.columns
        self.outcome_encoded_columns = outcome_enc.columns

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _sample(count: int) -> Tuple:
            # Static generation
            if self.static_model is None:
                static_enc = pd.DataFrame(np.zeros((count, 0)))
            else:
                static_enc = self.static_model.generate(count).numpy()

            # Temporal generation
            temporal_enc_raw = self.temporal_model.sample(count)
            temporal_enc = []
            for item in temporal_enc_raw:
                temporal_enc.append(
                    pd.DataFrame(item, columns=self.temporal_encoded_columns)
                )

            # Outcome generation
            outcome_enc = pd.DataFrame(
                self.outcome_model.predict(
                    np.asarray(static_enc), np.asarray(temporal_enc)
                ),
                columns=self.outcome_encoded_columns,
            )

            static_raw, temporal_raw = self.temporal_encoder.inverse_transform(
                static_enc, temporal_enc
            )
            outcome_raw = self.outcome_encoder.inverse_transform(outcome_enc)

            temporal = []
            for item in temporal_raw:
                temporal.append(
                    pd.DataFrame(item, columns=self.data_info["temporal_features"])
                )
            outcome = pd.DataFrame(
                outcome_raw, columns=self.data_info["outcome_features"]
            )
            static = pd.DataFrame(static_raw, columns=self.data_info["static_features"])

            return static, temporal, outcome

        return self._safe_generate_time_series(_sample, count, syn_schema)


plugin = FourierFlowsPlugin
