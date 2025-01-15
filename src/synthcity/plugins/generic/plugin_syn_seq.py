# File: plugin_syn_seq.py

from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

# synthcity absolute
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.dataloader import DataLoader, Syn_SeqDataLoader
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.core.constraints import Constraints

# Our sequential aggregator
from synthcity.plugins.core.models.syn_seq.syn_seq import Syn_Seq


class Syn_SeqPlugin(Plugin):
    """
    A plugin for column-by-column sequential synthesis (similar to R's 'synthpop').
    Internally, it delegates to the `Syn_Seq` aggregator for the actual
    column-by-column logic:

    1) We expect a Syn_SeqDataLoader, which organizes columns, applies
       Syn_SeqEncoder for special values, etc.
    2) The aggregator `.fit()` will encode the loader, partial-fit each splitted column.
    3) The aggregator `.generate()` will produce new synthetic data, decode to original columns.
    4) This plugin returns that synthesized data as another Syn_SeqDataLoader.

    Example usage:

        from synthcity.plugins.core.dataloader import Syn_SeqDataLoader
        from synthcity.plugins import Plugins

        # Suppose df is your raw dataset
        loader = Syn_SeqDataLoader(
            df,
            syn_order = [...],
            col_type = {...},
            special_value = {...},
        )

        # Build the plugin
        syn_model = Plugins().get("syn_seq")(
            random_state=42,
            default_first_method="swr",
            default_other_method="cart",
            strict=True,
            sampling_patience=500,
        )

        # Optionally define user_custom
        user_custom = {
            "syn_order": [...],
            "method": {"bp":"polyreg"},
            "variable_selection": {
                "target": ["bp","bmi","age", ...]
            }
        }

        # Fit
        syn_model.fit(loader, user_custom=user_custom)

        # Synthesize
        constraints = {
            "target": [
                ("bmi", ">", 0.15),
                ("target", ">", 0)
            ]
        }
        synthetic_loader = syn_model.generate(count=1000, constraints=constraints)
        synthetic_df = synthetic_loader.dataframe()
    """

    @staticmethod
    def name() -> str:
        return "syn_seq"

    @staticmethod
    def type() -> str:
        return "syn_seq"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> list:
        # No hyperparameters to tune by default
        return []

    def __init__(
        self,
        random_state: int = 0,
        default_first_method: str = "swr",
        default_other_method: str = "cart",
        strict: bool = True,
        sampling_patience: int = 100,
        **kwargs: Any,
    ):
        """
        Args:
            random_state: seed for reproducibility
            default_first_method: fallback method for the first column (if not user-specified)
            default_other_method: fallback for subsequent columns
            strict: if True => repeated attempts to meet constraints
            sampling_patience: maximum number of tries in strict mode
            **kwargs: forwarded to the Plugin base class (e.g., plugin name, etc.)
        """
        super().__init__(random_state=random_state, **kwargs)

        # Build our aggregator with the relevant arguments
        self._aggregator = Syn_Seq(
            random_state=random_state,
            strict=strict,
            sampling_patience=sampling_patience,
            default_first_method=default_first_method,
            default_other_method=default_other_method,
        )
        self._model_trained = False

    def _fit(
        self,
        X: DataLoader,
        user_custom: Optional[dict] = None,
        *args: Any,
        **kwargs: Any
    ) -> "Syn_SeqPlugin":
        """
        Fit the aggregator using a Syn_SeqDataLoader.

        Args:
            X: Must be a Syn_SeqDataLoader containing the raw data, syn_order, col_type, etc.
            user_custom: a dictionary that can specify:
                - "syn_order" : list of columns in desired order
                - "method" : { column_name: method_name, ... }
                - "variable_selection" : { target_col : [predictor_cols] }
            *args, **kwargs: additional parameters (if needed)

        Raises:
            TypeError if X is not a Syn_SeqDataLoader
        """
        if not isinstance(X, Syn_SeqDataLoader):
            raise TypeError("Syn_SeqPlugin expects a Syn_SeqDataLoader for sequential usage.")

        # aggregator fit => merges user_custom into loader => encode => partial-fit
        self._aggregator.fit(
            loader=X,
            user_custom=user_custom,
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
        X: Optional[DataLoader] = None,
        **kwargs: Any
    ) -> DataLoader:
        """
        Generate synthetic data as a DataLoader (Syn_SeqDataLoader).
        We pass optional constraints to aggregator (which can be dict or Constraints object).

        Args:
            count: number of rows to generate
            syn_schema: a Schema object (unused here, but required by base)
            constraints: a constraints dictionary or Constraints object
            X: a DataLoader. Typically the same we used for fit. If None, aggregator can't decode.
            **kwargs: additional parameters for aggregator.generate

        Returns:
            A Syn_SeqDataLoader containing the synthetic data.
        """
        if not self._model_trained:
            raise RuntimeError("Must fit Syn_SeqPlugin before generating data.")

        if X is None:
            raise ValueError("Syn_SeqPlugin.generate requires a DataLoader reference for decoding.")

        # aggregator.generate => returns a pd.DataFrame
        df_synth = self._aggregator.generate(
            count=count,
            encoded_loader=X,  # pass the same (or similar) loader
            constraints=constraints,
            **kwargs
        )
        # We want to return a Syn_SeqDataLoader (like the aggregator once did).
        # We can just decorate it with X, so the user can .dataframe() or .decode() further.
        return X.decorate(df_synth)

plugin = Syn_SeqPlugin