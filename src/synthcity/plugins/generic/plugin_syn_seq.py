# File: plugin_syn_seq.py

from pathlib import Path
from typing import Any, List, Optional, Union, Dict
import pandas as pd

from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.dataloader import DataLoader, create_from_info
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.core.serializable import Serializable
from synthcity.utils.reproducibility import enable_reproducible_results

# local aggregator
from synthcity.plugins.core.models.syn_seq.syn_seq import Syn_Seq


class Syn_SeqPlugin(Plugin):
    """
    The 'syn_seq' plugin for synthetic data using sequential-synthesis logic (non-deep-learning).
      - Wraps the Syn_Seq aggregator in the standard Plugin interface.
    """

    @staticmethod
    def name() -> str:
        return "syn_seq"

    @staticmethod
    def type() -> str:
        return "syn_seq"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        # Provide any hyperparameter distributions you want to tune. For now, none => return []
        return []

    @validate_arguments
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
        """
        Args:
            random_state: random seed
            sampling_patience: max tries to meet constraints
            strict: if True => discard rows that do not meet constraints
            workspace: local caching path
            compress_dataset: not used here
            sampling_strategy: default 'marginal'
        """
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
        enable_reproducible_results(self.random_state)

        self.model = Syn_Seq(
            random_state=self.random_state,
            strict=self.strict,
            sampling_patience=self.sampling_patience,
        )
        self.model.fit(X, *args, **kwargs)
        return self

    def _generate(
        self,
        count: int,
        syn_schema: Schema,
        **kwargs: Any
    ) -> DataLoader:
        """
        By default, parent Plugin's .generate() calls _generate(...). But here we do not rely on domain constraints,
        we let the aggregator do the generation. Then we adapt dtypes, etc.
        """
        # Actually we'll raise if used, because we override generate() below with custom logic.
        raise NotImplementedError(
            "Syn_SeqPlugin uses a custom .generate() that bypasses the parent's domain constraints."
        )

    @validate_arguments
    def generate(
        self,
        count: Optional[int] = None,
        constraints: Optional[Constraints] = None,
        random_state: Optional[int] = None,
        **kwargs: Any
    ) -> DataLoader:
        if not self.fitted or not self.model:
            raise RuntimeError("Must fit() plugin before calling generate().")

        if count is None:
            count = self.data_info["len"]

        if random_state is not None:
            import numpy as np
            np.random.seed(random_state)

        # For rules logic
        rules = kwargs.pop("rules", None)

        # aggregator-based generation
        raw_df = self.model.generate(nrows=count, rules=rules)
        X_syn = create_from_info(raw_df, self.data_info)

        # apply constraints if strict
        if constraints is not None and self.strict:
            X_syn = X_syn.match(constraints)

        # decode data (revert label encoding, date offsets, etc.)
        if X_syn.is_tabular() and self._data_encoders is not None:
            X_syn = X_syn.decode(self._data_encoders)

        return X_syn


# Required for the plugin auto-loader
plugin = Syn_SeqPlugin
