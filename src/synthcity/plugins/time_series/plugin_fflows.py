"""
Reference: "Generative Time-series Modeling with Fourier Flows", Ahmed Alaa, Alex Chan, and Mihaela van der Schaar.
"""
# stdlib
from pathlib import Path
from typing import Any, List, Tuple

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
from synthcity.plugins.core.models.ts_model import TimeSeriesModel
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.generic import GenericPlugins
from synthcity.utils.constants import DEVICE


class FourierFlowsPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.time_series.plugin_fflows.FourierFlowsPlugin
        :parts: 1


    Synthetic time series generation using FourierFlows.

    Args:
        n_iter: int
            Number of training iterations
        batch_size: int
            Batch size
        lr: float
            Learning rate
        n_iter_print: int
            Number of iterations to print the validation loss
        n_units_hidden: int
            Number of hidden nodes
        n_flows: int
            Number of flows to use(default = 10)
        FFT: bool
            Use Fourier transform(default = True)
        flip: bool
            Flip the data in the SpectralFilter
        normalize: bool
            Scale the data(default = False)
        static_model: str = "ctgan",
            The model to use for generating the static data.
        device: Any = DEVICE
            torch device to use for training(cpu/cuda)
        encoder_max_clusters: int = 10
            Number of clusters used for tabular encoding
        # Core Plugin arguments
        workspace: Path.
            Optional Path for caching intermediary results.
        compress_dataset: bool. Default = False.
            Drop redundant features before training the generator.
        sampling_patience: int.
            Max inference iterations to wait for the generated data to match the training schema.

    Example:
        >>> from synthcity.plugins import Plugins
        >>> from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
        >>> from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
        >>> static, temporal, horizons, outcome = GoogleStocksDataloader().load()
        >>> loader = TimeSeriesDataLoader(
        >>>             temporal_data=temporal,
        >>>             observation_times=horizons,
        >>>             static_data=static,
        >>>             outcome=outcome,
        >>> )
        >>>
        >>> plugin = Plugins().get("fflows", n_iter = 50)
        >>> plugin.fit(loader)
        >>>
        >>> plugin.generate(count = 10)


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
        # core plugin arguments
        random_state: int = 0,
        workspace: Path = Path("workspace"),
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
        self.static_model_name = static_model
        self.device = device

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

        self.static_model = GenericPlugins().get(
            self.static_model_name, device=self.device
        )

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
        if X.type() not in ["time_series", "time_series_survival"]:
            raise ValueError(f"Invalid data type = {X.type()}")

        if X.type() == "time_series":
            static, temporal, observation_times, outcome = X.unpack(pad=True)
        elif X.type() == "time_series_survival":
            static, temporal, observation_times, T, E = X.unpack(pad=True)
            outcome = pd.concat([pd.Series(T), pd.Series(E)], axis=1)
            outcome.columns = ["time_to_event", "event"]

        # Train static generator
        self.temporal_encoder.fit_temporal(temporal, observation_times)
        (
            temporal_enc,
            observation_times_enc,
        ) = self.temporal_encoder.transform_temporal(temporal, observation_times)

        static_data_with_horizons = np.concatenate(
            [np.asarray(static), np.asarray(observation_times)], axis=1
        )
        self.static_model.fit(pd.DataFrame(static_data_with_horizons))

        # Train temporal generator
        self.temporal_model.fit(temporal_enc, **self.train_args)

        # Train outcome generator
        self.outcome_encoder.fit(outcome)
        outcome_enc = self.outcome_encoder.transform(outcome)

        self.outcome_model = TimeSeriesModel(
            task_type="regression",
            n_static_units_in=static.shape[-1],
            n_temporal_units_in=temporal[0].shape[-1],
            n_temporal_window=temporal[0].shape[0],
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
            np.asarray(static),
            np.asarray(temporal),
            np.asarray(observation_times),
            np.asarray(outcome_enc),
        )

        self.temporal_encoded_columns = temporal_enc[0].columns
        self.outcome_encoded_columns = outcome_enc.columns
        self.static_columns = static.columns

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _sample(count: int) -> Tuple:
            # Static generation
            static_data_with_horizons = self.static_model.generate(count).numpy()
            static = pd.DataFrame(
                static_data_with_horizons[:, : len(self.static_columns)],
                columns=self.static_columns,
            )
            observation_times_enc = static_data_with_horizons[
                :, len(self.static_columns) :
            ]

            # Temporal generation
            temporal_enc_raw = self.temporal_model.sample(count)
            temporal_enc = []
            for item in temporal_enc_raw:
                temporal_enc.append(
                    pd.DataFrame(item, columns=self.temporal_encoded_columns)
                )

            # Decoding
            (
                temporal_raw,
                observation_times,
            ) = self.temporal_encoder.inverse_transform_temporal(
                temporal_enc, observation_times_enc.tolist()
            )

            temporal = []
            for item in temporal_raw:
                temporal.append(
                    pd.DataFrame(item, columns=self.data_info["temporal_features"])
                )

            # Outcome generation
            outcome_enc = pd.DataFrame(
                self.outcome_model.predict(
                    np.asarray(static),
                    np.asarray(temporal_raw),
                    np.asarray(observation_times),
                ),
                columns=self.outcome_encoded_columns,
            )
            outcome_raw = self.outcome_encoder.inverse_transform(outcome_enc)
            outcome = pd.DataFrame(
                outcome_raw, columns=self.data_info["outcome_features"]
            )
            static = pd.DataFrame(static, columns=self.data_info["static_features"])

            if self.data_info["data_type"] == "time_series":
                return static, temporal, observation_times, outcome
            elif self.data_info["data_type"] == "time_series_survival":
                return (
                    static,
                    temporal,
                    observation_times,
                    outcome[self.data_info["time_to_event_column"]],
                    outcome[self.data_info["event_column"]],
                )
            else:
                raise RuntimeError("unknow data type")

        return self._safe_generate_time_series(_sample, count, syn_schema)


plugin = FourierFlowsPlugin
