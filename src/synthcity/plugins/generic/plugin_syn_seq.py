# File: plugin_syn_seq.py

from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

# synthcity absolute
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.dataloader import DataLoader, Syn_SeqDataLoader
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.core.constraints import Constraints

# local aggregator
from synthcity.plugins.core.models.syn_seq.syn_seq import Syn_Seq


class Syn_SeqPlugin(Plugin):
    """
    A plugin for a sequential (column-by-column) synthetic data approach,
    mirroring R's 'synthpop'. Internally, it delegates to the `Syn_Seq`
    aggregator from syn_seq.py for the actual column-by-column (sequential) logic.

    - This is intended for sequential regression style: each column is modeled
      in the order, using prior columns as predictors.
    - We rely on the custom `Syn_SeqDataLoader`, which organizes columns,
      applies a Syn_SeqEncoder, etc.
    - The aggregator's generate(...) returns a Syn_SeqDataLoader as well.

    Basic usage:

        # Suppose we have df, and we wrap it in a Syn_SeqDataLoader:
        loader = Syn_SeqDataLoader(
            data = df, 
            syn_order = [...],
            col_type = {...},
            special_value = {...},
        )

        # Build plugin
        syn_model = Syn_SeqPlugin(
            random_state=42,
            default_first_method="SWR", 
            default_other_method="CART",
            # optionally strict=True, sampling_patience=500, ...
        )

        # Fit with per-column method
        methods = ["SWR"] + ["CART"]*(len(loader.columns)-1)
        var_sel = {"N2": ["C1","C2"], "N1":["C1","C2","N2"]}

        syn_model.fit(loader, method=methods, variable_selection=var_sel)

        # Now generate
        synthetic_data_loader = syn_model.generate(count=100)
        synthetic_df = synthetic_data_loader.dataframe()

        # If you need constraints:
        constraints = {
          "N1": [">", 100],
          "C2": ["in", ["A","B"]]
        }
        synthetic_data_loader = syn_model.generate(count=100, constraints=constraints)
        # Above remains a Syn_SeqDataLoader
    """

    @staticmethod
    def name() -> str:
        return "syn_seq"

    @staticmethod
    def type() -> str:
        return "syn_seq"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> list:
        # No tunable hyperparameters for now
        return []

    def __init__(
        self,
        random_state: int = 0,
        default_first_method: str = "SWR",
        default_other_method: str = "CART",
        **kwargs: Any,
    ):
        """
        Args:
            random_state: For reproducibility.
            default_first_method: fallback method for the first column if user doesn't override.
            default_other_method: fallback method for subsequent columns.
            **kwargs: forwarded to Plugin(...) => can contain strict, sampling_patience, etc.
        """
        super().__init__(random_state=random_state, **kwargs)

        # We hold a Syn_Seq aggregator. This aggregator does the real sequential logic:
        self._aggregator = Syn_Seq(
            random_state=random_state,
            default_first_method=default_first_method,
            default_other_method=default_other_method,
            strict=self.strict,
            sampling_patience=self.sampling_patience,
        )
        self._model_trained = False

    def _fit(
        self,
        X: DataLoader,
        method: Optional[List[str]] = None,
        variable_selection: Optional[Dict[str, List[str]]] = None,
        *args: Any,
        **kwargs: Any
    ) -> "Syn_SeqPlugin":
        """
        We expect X to be a Syn_SeqDataLoader for sequential usage.
        We pass 'method' and 'variable_selection' to the aggregator.
        """
        if not isinstance(X, Syn_SeqDataLoader):
            raise TypeError("Syn_SeqPlugin expects a Syn_SeqDataLoader for sequential usage.")

        self._aggregator.fit(
            loader=X,
            method=method,
            variable_selection=variable_selection,
            *args,
            **kwargs
        )
        self._model_trained = True
        return self

    def _generate(
        self,
        count: int,
        syn_schema: Schema,
        constraints: Optional[Constraints] = None,
        **kwargs: Any
    ) -> DataLoader:
        """
        Create synthetic data as a Syn_SeqDataLoader. We pass optional constraints
        (which might be a dictionary or a Constraints object) to aggregator.
        """
        if not self._model_trained:
            raise RuntimeError("Must fit Syn_SeqPlugin before generating data.")

        return self._aggregator.generate(
            count=count,
            constraint=constraints,
            **kwargs
        )

plugin = Syn_SeqPlugin
