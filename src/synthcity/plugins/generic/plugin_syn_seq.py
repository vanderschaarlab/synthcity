# File: plugins/generic/plugin_syn_seq.py

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from random import sample
import numpy as np

# synthcity absolute
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader, Syn_SeqDataLoader
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.models.syn_seq_encoder import Syn_SeqEncoder

class Syn_SeqPlugin(Plugin):
    """
    A plugin for a sequential (column-by-column) synthetic data approach,
    akin to R's 'synthpop' but in Python. It integrates:

    1) A specialized DataLoader (Syn_SeqDataLoader) that reorders columns, handles special values, etc.
    2) An encoder (Syn_SeqEncoder) that splits numeric columns, marks date columns, etc.
    3) A simple column-by-column modeling approach for synthetic generation.
    4) Constraint handling (post-hoc '=' assignment or hooking into parent `_safe_generate` if strict is True).

    Usage example:
        loader = Syn_SeqDataLoader(
            data = my_dataframe,
            syn_order = ["C2", "C1", "N2", "N1", "D1"],
            special_value = {"N2": [-888, 888], "D1": [0]},
            col_type = {"C2":"category", "C1":"category", "N2":"numeric", "N1":"numeric", "D1":"date"}
        )

        syn_model = Plugins().get("syn_seq")  # or Syn_SeqPlugin()
        syn_model.fit(
            loader,
            method = ["SWR", "CART", "CART", "pmm", "CART"],
            variable_selection = {
                "N1": ["C2", "C1"],
                "D1": ["C2", "C1", "N2"]
            }
        )

        constraint = {
            "N1": [">", 125],
            # "C1": ["in", ["AAA","BBB"]],
            # "D1": ["=", "2020-01-01"]
        }

        syn_df = syn_model.generate(count=100, constraint=constraint).dataframe()
    """

    @staticmethod
    def name() -> str:
        return "syn_seq"

    @staticmethod
    def type() -> str:
        return "syn_seq"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> list:
        # No tunable hyperparameters here; placeholders for future extension.
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
            random_state: Fixes seed for reproducibility.
            default_first_method: method for the first column if user doesn't override. e.g. 'SWR'.
            default_other_method: method for subsequent columns if user doesn't override. e.g. 'CART'.
            **kwargs: passes through to Plugin base (e.g. strict, sampling_patience, workspace, etc.)
        """
        super().__init__(random_state=random_state, **kwargs)

        # Where we store each column's fitted "model": { col : { "method":..., "predictors":..., "model":... } }
        self._column_models: Dict[str, dict] = {}

        # For user configuration
        self.method_list: List[str] = []
        self.variable_selection: Dict[str, Union[List[str], List[int]]] = {}

        # If user doesn't supply a method for each column, we fill with these defaults:
        self.default_first_method = default_first_method
        self.default_other_method = default_other_method

        # Encoders, training state
        self._encoders: Dict[str, Any] = {}
        self._model_trained = False

    # ----------------------------------------------------------------
    # OVERRIDES FROM Plugin
    # ----------------------------------------------------------------

    def _fit(
        self,
        X: DataLoader,
        method: Optional[List[str]] = None,
        variable_selection: Optional[Dict[str, List[str]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> "Syn_SeqPlugin":
        """
        Internal training logic, after parent's .fit sets up schema, etc.

        Steps:
          - Possibly expand user-supplied method_list so each column has a method
          - Build or adapt variable_selection matrix
          - Print out final assignments
          - "Train" a minimal column model
        """
        if not isinstance(X, Syn_SeqDataLoader):
            raise TypeError("Syn_SeqPlugin expects a Syn_SeqDataLoader after .encode()")

        df = X.dataframe()
        col_list = list(df.columns)
        n_cols = len(col_list)

        # capture user parameters if they exist
        if method is not None:
            self.method_list = method
        if variable_selection is not None:
            self.variable_selection = variable_selection

        # Expand/assign final column methods
        final_methods = []
        for i, col in enumerate(col_list):
            if i < len(self.method_list):
                final_methods.append(self.method_list[i])
            else:
                # fallback
                fallback = self.default_first_method if i == 0 else self.default_other_method
                final_methods.append(fallback)

        # Print final method assignments
        print("[INFO] Final column method assignment:")
        for col, m in zip(col_list, final_methods):
            print(f"  {col}: {m}")

        # Build a default variable_selection matrix if user doesn't supply one
        vs_matrix = pd.DataFrame(0, index=col_list, columns=col_list)
        for i in range(n_cols):
            vs_matrix.iloc[i, :i] = 1

        # overlay user-specified variable_selection
        for col, preds in self.variable_selection.items():
            if col not in vs_matrix.index:
                continue
            vs_matrix.loc[col, :] = 0
            for p in preds:
                if p in vs_matrix.columns:
                    vs_matrix.loc[col, p] = 1

        print("[INFO] Final variable selection matrix:")
        print(vs_matrix)

        # "Train" each column
        self._column_models.clear()
        for i, col in enumerate(col_list):
            chosen_method = final_methods[i]
            predictor_cols = vs_matrix.columns[(vs_matrix.loc[col] == 1)].tolist()

            model_info = self._train_column_model(
                df,
                target_col=col,
                predictor_cols=predictor_cols,
                method=chosen_method,
            )
            self._column_models[col] = {
                "method": chosen_method,
                "predictors": predictor_cols,
                "model": model_info,
            }

        self._model_trained = True
        return self

    def _generate(
        self,
        count: int,
        syn_schema: Schema,
        **kwargs: Any
    ) -> DataLoader:
        """
        Our core row-by-row synthetic generation logic, invoked by parent's `_safe_generate`.

        Steps:
          - Create an empty DF of size 'count'
          - For each column in order, sample values from the column model
          - Return as a GenericDataLoader
        """
        col_list = list(self._column_models.keys())
        syn_df = pd.DataFrame(index=range(count))

        for col in col_list:
            info = self._column_models[col]
            method = info["method"]
            preds = info["predictors"]
            model_obj = info["model"]

            new_values = self._generate_for_column(
                count,
                col,
                method,
                preds,
                model_obj,
                partial_df=syn_df,
            )
            syn_df[col] = new_values

        return GenericDataLoader(syn_df)

 