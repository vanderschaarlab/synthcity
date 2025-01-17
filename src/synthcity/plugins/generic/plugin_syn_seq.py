# File: plugin_syn_seq.py

from pathlib import Path
from typing import Any, List, Optional, Union, Dict
import pandas as pd

from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.dataloader import DataLoader, Syn_SeqDataLoader
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.models.syn_seq.syn_seq import Syn_Seq
from synthcity.plugins.core.constraints import Constraints

class Syn_SeqPlugin(Plugin):
    """
    A plugin wrapping the 'Syn_Seq' aggregator in the synthcity Plugin interface.
    """

    @staticmethod
    def name() -> str:
        return "syn_seq"

    @staticmethod
    def type() -> str:
        return "syn_seq"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        # Return an empty list for now, or add distributions for tuning your hyperparams
        return []

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        random_state: int = 0,
        sampling_patience: int = 100,
        strict: bool = True,
        workspace: Path = Path("workspace"),
        compress_dataset: bool = False,
        sampling_strategy: str = "marginal",
        **kwargs: Any
    ) -> None:
        # Pass these to the base class Plugin
        super().__init__(
            random_state=random_state,
            sampling_patience=sampling_patience,
            strict=strict,
            workspace=workspace,
            compress_dataset=compress_dataset,
            sampling_strategy=sampling_strategy,
        )
        self.model: Optional[Syn_Seq] = None

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "Syn_SeqPlugin":
        """
        Internally call Syn_Seq.fit(...)
        """
        # Create aggregator
        self.model = Syn_Seq(
            random_state=self.random_state,
            strict=self.strict,
            sampling_patience=self.sampling_patience,
        )
        # Fit aggregator
        self.model.fit(X, *args, **kwargs)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> DataLoader:
        """
        Use Syn_Seq.generate(...) to produce the raw DataFrame,
        then convert it to a Syn_SeqDataLoader with the same 'info' as training data (except we just guess).
        We can also handle user-provided 'rules' from kwargs if needed.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
        rules = kwargs.pop("rules", None)

        # We'll do the standard "safe_generate" approach
        def gen_cbk(num_samples: int, **inner_kwargs):
            # returns a raw pd.DataFrame
            return self.model.generate(nrows=num_samples, rules=rules)

        # If your data is tabular, we can do the usual _safe_generate
        if self.data_info["data_type"] == "syn_seq":
            # We do self._safe_generate with gen_cbk
            return self._safe_generate(gen_cbk, count, syn_schema, **kwargs)
        elif self.data_info["data_type"] in ["time_series", "time_series_survival"]:
            return self._safe_generate_time_series(gen_cbk, count, syn_schema, **kwargs)
        else:
            return self._safe_generate(gen_cbk, count, syn_schema, **kwargs)

plugin = Syn_SeqPlugin