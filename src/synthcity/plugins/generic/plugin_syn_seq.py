from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from pydantic import validate_arguments

from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.dataloader import DataLoader, Syn_SeqDataLoader, create_from_info
from synthcity.plugins.core.models.syn_seq.syn_seq import Syn_Seq
from synthcity.plugins.core.schema import Schema


class Syn_SeqPlugin(Plugin):
    @staticmethod
    def name() -> str:
        return "syn_seq"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List:
        # There are no tunable hyperparameters for Syn_Seq.
        return []

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        sampling_patience: int = 100,
        random_state: int = 0,
        compress_dataset: bool = False,
        sampling_strategy: str = "marginal",
        **kwargs: Any
    ) -> None:
        super().__init__(
            random_state=random_state,
            sampling_patience=sampling_patience,
            compress_dataset=compress_dataset,
            sampling_strategy=sampling_strategy,
        )
        self.model: Optional[Syn_Seq] = None

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "Syn_SeqPlugin":
        # If X is a plain DataFrame, wrap it in our Syn_SeqDataLoader.
        if isinstance(X, pd.DataFrame):
            X = Syn_SeqDataLoader(X)
        # Initialize the Syn_Seq aggregator and train it column‐by‐column.
        self.model = Syn_Seq(
            random_state=self.random_state,
            sampling_patience=self.sampling_patience,
        )
        # No extra call to X.encode() is needed here because we want to keep the original data form.
        self.model.fit_col(X, *args, **kwargs)
        self.data_info = X.info()
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> DataLoader:
        if self.model is None:
            raise RuntimeError("The model must be fitted before generating data.")
        # Generate synthetic data using the Syn_Seq aggregator.
        df_syn = self.model.generate_col(count, **kwargs)
        # Adapt the generated DataFrame to the schema (i.e. ensure data types match).
        df_syn = syn_schema.adapt_dtypes(df_syn)
        # Create a DataLoader from the synthetic DataFrame using the stored data_info.
        data_syn = create_from_info(df_syn, self.data_info)
        return data_syn


plugin = Syn_SeqPlugin
