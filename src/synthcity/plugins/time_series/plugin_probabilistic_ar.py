# stdlib
from typing import Any, List, Tuple

# third party
import pandas as pd
from deepecho import PARModel

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader, create_from_info
from synthcity.plugins.core.distribution import Distribution, IntegerDistribution
from synthcity.plugins.core.models.tabular_encoder import TabularEncoder
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE


class ProbabilisticAutoregressivePlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.time_series.plugin_probabilistic_ar.ProbabilisticAutoregressivePlugin
        :parts: 1


    Synthetic time series generation using Probabilistic Autoregressive models.

    Args:
        n_iter: int
            Maximum number of iterations in the Generator.
        sample_size: int
            The number of times to sample (before choosing and returning the sample which maximizes the likelihood). Defaults to 1.
        device: DEVICE
            torch device to use for training(cpu/cuda)
        encoder_max_clusters: int
            Number of clusters used for tabular encoding

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
        >>> plugin = Plugins().get("probabilistic_ar", n_iter = 50)
        >>> plugin.fit(loader)
        >>>
        >>> plugin.generate(count = 10)

    Reference: https://github.com/sdv-dev/DeepEcho
    """

    def __init__(
        self,
        n_iter: int = 200,
        sample_size: int = 1,
        device: Any = DEVICE,
        encoder_max_clusters: int = 10,
        **kwargs: Any
    ) -> None:
        super().__init__()

        self.model = PARModel(epochs=n_iter, sample_size=sample_size, verbose=False)
        self.encoder = TabularEncoder(
            max_clusters=encoder_max_clusters, whitelist=["seq_id", "seq_time_id"]
        )

    @staticmethod
    def name() -> str:
        return "probabilistic_ar"

    @staticmethod
    def type() -> str:
        return "time_series"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="n_iter", low=100, high=1000, step=100),
            IntegerDistribution(name="sample_size", low=1, high=10),
        ]

    def _fit(
        self, X: DataLoader, *args: Any, **kwargs: Any
    ) -> "ProbabilisticAutoregressivePlugin":
        if X.type() not in ["time_series", "time_series_survival"]:
            raise ValueError("Invalid data type = {X.type()}")

        if X.type() == "time_series":
            static, temporal, observation_times, outcome = X.unpack(pad=True)
        elif X.type() == "time_series_survival":
            static, temporal, observation_times, T, E = X.unpack(pad=True)
            outcome = pd.concat([pd.Series(T), pd.Series(E)], axis=1)
            outcome.columns = ["time_to_event", "event"]

        seq_df = X.dataframe().copy()
        self.info = X.info()

        id_col = self.info["seq_id_feature"]
        time_col = self.info["seq_time_id_feature"]

        seq_df["par_bkp_time"] = seq_df[time_col]

        # Train encoder
        seq_enc_df = self.encoder.fit_transform(seq_df)

        # Train the static and temporal generator
        self.model.fit(
            data=seq_enc_df,
            entity_columns=[id_col],
            context_columns=[],
            sequence_index=time_col,
        )

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _sample(count: int) -> Tuple:
            # Static and Temporal generation
            data = self.model.sample(num_entities=count)
            data = self.encoder.inverse_transform(data)
            time_col = self.info["seq_time_id_feature"]
            bkp_time_col = "par_bkp_time"

            # Decoding
            data[time_col] = data[bkp_time_col]
            data = data.drop(columns=[bkp_time_col])

            loader = create_from_info(data, self.info)

            return loader.unpack()

        return self._safe_generate_time_series(_sample, count, syn_schema)


plugin = ProbabilisticAutoregressivePlugin
