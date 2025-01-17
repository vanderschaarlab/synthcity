# File: plugin_syn_seq.py

from pathlib import Path
from typing import Any, List, Optional, Union, Dict
import pandas as pd

from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.dataloader import (
    DataLoader,
    create_from_info,
)
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.core.models.syn_seq.syn_seq import Syn_Seq


class Syn_SeqPlugin(Plugin):
    """
    A plugin wrapping the 'Syn_Seq' aggregator in the synthcity Plugin interface.

    **Key Features**:
      - We override .generate() to bypass the parent's domain-based constraints
        (i.e. skipping `Schema.from_constraints(...))`.
      - We define a minimal `_generate(...)` method so the class is no longer abstract
        (the base Plugin requires `_generate(...)` to exist).
      - After generating, we also decode the data if `self._data_encoders` exists. 
        This step is essential so the final returned DataLoader matches the original 
        data format (e.g. reversing label-encodings, etc.).

    Where to put the decoder:
      - Notice in our `generate(...)` method, after we call
        `create_from_info(raw_df, self.data_info)`, we get a DataLoader with 
        the same column structure but still possibly encoded. 
        If we want the original format (column dtypes, special values, etc.), 
        we do:
           if X_syn.is_tabular() and self._data_encoders is not None:
               X_syn = X_syn.decode(self._data_encoders)
        Then the returned DataLoader .dataframe() will be in the original format.

    """

    @staticmethod
    def name() -> str:
        return "syn_seq"

    @staticmethod
    def type() -> str:
        return "syn_seq"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        # Provide any hyperparameter distributions you want to tune
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
        """
        Args:
            random_state: int
                Random seed for reproducibility.
            sampling_patience: int
                Maximum iterations for sampling if we must meet constraints.
            strict: bool
                Whether to drop rows that do not meet constraints (True) 
                or keep them (False).
            workspace: Path
                Caching path.
            compress_dataset: bool
                If True, the plugin can optionally compress the dataset 
                before training (to remove redundancy).
            sampling_strategy: str
                Usually "marginal" or "uniform" (internal).
            **kwargs:
                Additional plugin arguments if needed.
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

    # ---------------------------------------------------------------------
    # 1) Fit the aggregator
    # ---------------------------------------------------------------------
    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "Syn_SeqPlugin":
        """
        Internal method for fitting the aggregator to the data
        """
        # Build aggregator
        self.model = Syn_Seq(
            random_state=self.random_state,
            strict=self.strict,
            sampling_patience=self.sampling_patience,
        )
        # Fit aggregator
        self.model.fit(X, *args, **kwargs)
        return self

    # ---------------------------------------------------------------------
    # 2) Minimal _generate(...) to satisfy abstract requirement
    #    (We do not actually use this code path in this plugin.)
    # ---------------------------------------------------------------------
    def _generate(
        self,
        count: int,
        syn_schema: Schema,
        **kwargs: Any
    ) -> DataLoader:
        """
        The parent's .generate(...) calls ._generate(...) by default. 
        But here we override .generate(...) entirely, so this is effectively unused.
        We raise an error or you can choose to do a fallback generation 
        if you prefer.
        """
        raise NotImplementedError(
            "Syn_SeqPlugin uses a custom .generate() that bypasses the parent's domain constraints."
        )

    # ---------------------------------------------------------------------
    # 3) Fully override .generate(...) to skip from_constraints + domain logic
    # ---------------------------------------------------------------------
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate(
        self,
        count: Optional[int] = None,
        constraints: Optional[Constraints] = None,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> DataLoader:
        """
        Custom generate method ignoring parent's ._generate(...) calls.

        Args:
            count: number of samples to generate. Defaults to length of training data.
            constraints: optional Constraints object. If `strict=True`, we will filter 
                         out violating rows after generation.
            random_state: optional int. If you want reproducible results, pass a new seed.
            rules: optional dict. e.g. rules = {
                'colname': [
                   ('feature1','>', 0.15),
                   ('colname','>', 0)
                ]
            }
            etc.
        Returns:
            A DataLoader with newly generated data, possibly decoded to original format.
        """
        if not self.fitted:
            raise RuntimeError("Must fit() plugin before calling generate().")

        if count is None:
            count = self.data_info["len"]  # fallback to training set size

        if random_state is not None:
            # re-seed if desired
            import numpy as np
            np.random.seed(random_state)

        # Extract 'rules' from kwargs if passed
        rules = kwargs.pop("rules", None)

        # Use our aggregator's direct generation
        raw_df = self.model.generate(nrows=count, rules=rules)
        # Convert raw DataFrame -> DataLoader with the same data_info as training set
        X_syn = create_from_info(raw_df, self.data_info)

        # (Optionally) apply constraints if we are strict
        if constraints is not None and self.strict:
            X_syn = X_syn.match(constraints)

        # (Optional) decode the data to restore original dtype/encoding
        # This is the recommended place to ensure returning the original data format:
        if X_syn.is_tabular() and self._data_encoders is not None:
            X_syn = X_syn.decode(self._data_encoders)

        return X_syn


# Required by the plugin loader
plugin = Syn_SeqPlugin
