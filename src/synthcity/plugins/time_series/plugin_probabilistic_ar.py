"""
Probabilistic autoregressive model
"""
# stdlib
from typing import Any, List, Tuple

# third party
import pandas as pd
from deepecho import PARModel

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader, TimeSeriesDataLoader
from synthcity.plugins.core.distribution import Distribution, IntegerDistribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE


class ProbabilisticAutoregressivePlugin(Plugin):
    """Synthetic time series generation using Probabilistic Autoregressive models.

    Args:
        n_iter: int
            Maximum number of iterations in the Generator.

    Example:
        >>> from synthcity.plugins import Plugins
        >>> from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
        >>> from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
        >>>
        >>> plugin = Plugins().get("probabilistic_ar")
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
        n_iter: int = 1000,
        sample_size: int = 1,
        device: Any = DEVICE,
        **kwargs: Any
    ) -> None:
        super().__init__()

        self.model = PARModel(epochs=n_iter, sample_size=sample_size, verbose=False)

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
        assert X.type() == "time_series"

        seq_df, info = X.sequential_view()
        self.info = info

        id_col = info["id_feature"]
        time_col = info["time_feature"]
        static_cols = info["static_features"]
        out_cols = info["outcome_features"]

        seq_df["par_bkp_time"] = seq_df[time_col]

        # Train the static and temporal generator
        self.model.fit(
            data=seq_df,
            entity_columns=[id_col],
            context_columns=static_cols + out_cols,
            sequence_index=time_col,
        )

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _sample(count: int) -> Tuple:
            # Static and Temporal generation
            data = self.model.sample(num_entities=count)
            time_col = self.info["time_feature"]
            bkp_time_col = "par_bkp_time"

            # Decoding
            data[time_col] = data[bkp_time_col]
            data = data.drop(columns=[bkp_time_col])
            loader = TimeSeriesDataLoader.from_sequential_view(data, self.info)

            return loader.unpack()

        return self._safe_generate_time_series(_sample, count, syn_schema)


plugin = ProbabilisticAutoregressivePlugin
