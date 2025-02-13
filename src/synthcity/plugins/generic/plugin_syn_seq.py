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
        # No tunable hyperparameters for syn_seq
        return []

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        sampling_patience: int = 100,
        strict: bool = True,
        random_state: int = 0,
        compress_dataset: bool = False,
        sampling_strategy: str = "marginal",
        **kwargs: Any
    ) -> None:
        super().__init__(
            random_state=random_state,
            sampling_patience=sampling_patience,
            strict=strict,
            compress_dataset=compress_dataset,
            sampling_strategy=sampling_strategy,
        )
        self.model: Optional[Syn_Seq] = None

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "Syn_SeqPlugin":
        # If a plain DataFrame is provided, wrap it in a Syn_SeqDataLoader
        if isinstance(X, pd.DataFrame):
            X = Syn_SeqDataLoader(X)
        # Initialize and train the Syn_Seq aggregator
        self.model = Syn_Seq(
            random_state=self.random_state,
            strict=self.strict,
            sampling_patience=self.sampling_patience,
        )
        self.model.fit_col(X, *args, **kwargs)
        # IMPORTANT: update the plugin's data_info using the loader’s updated info 
        # (which now includes the auto-injected “_cat” columns)
        self.data_info = X.info()
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> DataLoader:
        if self.model is None:
            raise RuntimeError("The model must be fitted before generating data.")
        # Generate synthetic data column-by-column
        df_syn = self.model.generate_col(count, **kwargs)
        # Adapt the data types according to the provided schema
        df_syn = syn_schema.adapt_dtypes(df_syn)
        # Create a DataLoader from the generated DataFrame using the updated data_info
        data_syn = create_from_info(df_syn, self.data_info)
        return data_syn


plugin = Syn_SeqPlugin
