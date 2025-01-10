# File: plugins/syn_seq/plugin_syn_seq.py

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from random import sample
import numpy as np

# synthcity absolute
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader
from synthcity.plugins.core.schema import Schema

# local references
from .syn_seq_dataloader import Syn_SeqDataLoader
from .syn_seq_encoder import Syn_SeqEncoder


class Syn_SeqPlugin(Plugin):
    """
    A plugin for a sequential (column-by-column) synthetic data approach,
    akin to R's `synthpop` but in Python. This plugin consolidates all logic:
    encoding, fitting methods per column, and generating new samples.

    Usage:
        loader = Syn_SeqDataLoader(...)
        syn_model = Plugins().get("syn_seq")

        syn_model.fit(
            loader,
            method = ["SWR", "CART", "CART", "pmm", "CART"],
            variable_selection = { "N1": ["C2", "C1"], "D1": ["C2", "C1", "N2"] }
        )

        constraints = {
          # (column, operation, value) in some dictionary format you like
          # or "N1": ["=", 999], "N1":[">", 10], etc.
        }

        syn_data = syn_model.generate(count=100, constraint=constraints)
        df_syn = syn_data.dataframe()
    """

    @staticmethod
    def name() -> str:
        return "syn_seq"

    @staticmethod
    def type() -> str:
        return "syn_seq"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> list:
        return []

    def __init__(
        self,
        random_state: int = 0,
        default_first_method: str = "SWR",
        default_other_method: str = "CART",
        **kwargs: Any
    ):
        """
        Args:
            default_first_method: if user doesn't override, the first column uses SWR.
            default_other_method: if user doesn't override, other columns use CART.
            **kwargs: passed to the base Plugin constructor (like strict=True, workspace=..., etc.)
        """
        super().__init__(random_state=random_state, **kwargs)

        # Storage for fitted results
        self._column_models: Dict[str, Any] = {}
        self.method_list: List[str] = []
        self.variable_selection: Dict[str, Union[List[str], List[int]]] = {}

        # Default methods
        self.default_first_method = default_first_method
        self.default_other_method = default_other_method

        # Encoders + tracking
        self._encoders: Dict[str, Any] = {}
        self._model_trained = False

    def fit(
        self,
        dataloader: Syn_SeqDataLoader,
        method: Optional[List[str]] = None,
        variable_selection: Optional[Dict[str, List[str]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> "Syn_SeqPlugin":
        """
        Fit the sequential synthesis model.

        Args:
            dataloader: Syn_SeqDataLoader with your real dataset.
            method: a list specifying the method for each column in the dataloader's syn_order.
                    e.g. ["SWR", "CART", "CART", "pmm", "CART"].
                    If absent or shorter than #cols, defaults used.
            variable_selection: dict controlling which predictor columns are used for each target col.
                                e.g. {"N1": ["C2","C1"], "D1": ["C2","C1","N2"]}

        Steps:
          1) We encode the data => returns a new loader + encoder.
          2) We store the user method array, user variable_selection.
          3) Print out the final method assignment & var-selection matrix for debugging.
          4) Actually train the column models (calls parent .fit => triggers _fit).
        """
        if not isinstance(dataloader, Syn_SeqDataLoader):
            raise TypeError("Syn_SeqPlugin expects a Syn_SeqDataLoader")

        # encode
        encoded_loader, encoders = dataloader.encode()
        df_enc = encoded_loader.dataframe()
        if df_enc.empty:
            raise ValueError("No data to train on in Syn_SeqPlugin.")

        self._encoders = encoders

        # store user info
        self.method_list = method or []
        self.variable_selection = variable_selection or {}

        # Now let the base Plugin handle schema etc. (calls _fit internally)
        super().fit(encoded_loader, *args, **kwargs)
        return self

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "Syn_SeqPlugin":
        """
        The real training logic after base .fit() sets up internal schema etc.
        We'll create a model for each column, in the order they appear in X.
        """
        df = X.dataframe()
        col_list = list(df.columns)

        # Possibly expand method_list to match # of columns
        final_methods = []
        for i, col in enumerate(col_list):
            if i < len(self.method_list):
                final_methods.append(self.method_list[i])
            else:
                # fallback
                final_methods.append(
                    self.default_first_method if i == 0 else self.default_other_method
                )

        # Print out the final method assignments for debugging
        print("[INFO] Final column method assignment:")
        for col, m in zip(col_list, final_methods):
            print(f"  {col}: {m}")

        # If we want to also show a final variable_selection matrix (like synthpop)
        # We create a matrix col_list x col_list of 0/1
        n = len(col_list)
        vs_matrix = pd.DataFrame(0, index=col_list, columns=col_list)
        # default: i-th row uses all j < i
        for i in range(n):
            vs_matrix.iloc[i, :i] = 1

        # incorporate user changes
        if self.variable_selection:
            for target_col, pred_cols in self.variable_selection.items():
                if target_col in vs_matrix.index:
                    # zero out that row
                    vs_matrix.loc[target_col, :] = 0
                    for pc in pred_cols:
                        if pc in vs_matrix.columns:
                            vs_matrix.loc[target_col, pc] = 1

        print("[INFO] Final variable selection matrix:")
        print(vs_matrix)

        # Train a small “model” for each column
        for i, col in enumerate(col_list):
            chosen_method = final_methods[i]
            # get predictor columns
            preds = vs_matrix.columns[(vs_matrix.loc[col] == 1)].tolist()

            # train a model
            model_info = self._train_column_model(
                df,
                target_col=col,
                predictor_cols=preds,
                method=chosen_method,
            )
            self._column_models[col] = {
                "method": chosen_method,
                "predictors": preds,
                "model": model_info,
            }

        self._model_trained = True
        return self

    def generate(
        self,
        count: int = 10,
        constraint: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Syn_SeqDataLoader:
        """
        Generate synthetic data for 'count' samples.

        If constraint is provided, we handle them in 2 ways:
          1) Hard constraints (like 'in', '>', etc.) are handled by the parent's `_safe_generate` logic,
             repeatedly sampling until the constraints are satisfied or out of patience.
          2) If we see e.g. 'col' : ["=", val], we interpret that as direct substitution after generation.

        Return a new Syn_SeqDataLoader with the final synthetic data.
        """
        if not self._model_trained:
            raise RuntimeError("Cannot generate. Must call fit() first.")

        # We'll store constraint in kwargs, so the parent's .generate => _generate => _safe_generate can see it
        # If there's a special '=' constraint, we do a simple post-processing approach in the method below.
        if constraint is not None:
            kwargs["syn_seq_constraint"] = constraint

        # parent's generate => calls _generate => we produce a DataLoader => it might do repeated tries
        gen_data: DataLoader = super().generate(count=count, *args, **kwargs)

        # decode if needed
        syn_df = gen_data.dataframe()
        if "syn_seq_encoder" in self._encoders:
            enc = self._encoders["syn_seq_encoder"]
            syn_df = enc.inverse_transform(syn_df)

        # now handle substitution constraints if any
        if constraint is not None:
            syn_df = self._apply_substitution_constraints(syn_df, constraint)

        # wrap into Syn_SeqDataLoader
        return Syn_SeqDataLoader(
            data=syn_df,
            syn_order=list(syn_df.columns),
        )

    def _generate(
        self,
        count: int,
        syn_schema: Schema,
        **kwargs: Any
    ) -> DataLoader:
        """
        Actually create synthetic rows. We'll do a simple column-by-column approach:
          1) For each column, call the stored model's sample function.
          2) Insert into partial synthetic DataFrame.
        Then we'll let the parent's `_safe_generate` handle iteration for constraints.
        """
        # We'll return a raw DataFrame, the parent plugin code wraps it in a DataLoader, tries constraints, etc.
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

    # ----------------------------------------------------------------
    # Helper: "train" a column
    # ----------------------------------------------------------------
    def _train_column_model(
        self,
        df: pd.DataFrame,
        target_col: str,
        predictor_cols: List[str],
        method: str,
    ) -> dict:
        """
        Minimalistic placeholder training code. 
        For real logic, plug in your CART, pmm, etc. 
        We'll store raw data or any other info we need.
        """
        model_info = {
            "predictors": predictor_cols,
            "target_data": df[target_col].values,  # store entire column
            "method": method,
        }
        return model_info

    # ----------------------------------------------------------------
    # Helper: generate synthetic values for one column
    # ----------------------------------------------------------------
    def _generate_for_column(
        self,
        count: int,
        col: str,
        method: str,
        predictor_cols: List[str],
        model_obj: dict,
        partial_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Simple sampling based on method. 
        Real "CART"/"pmm" etc. logic would be placed here.
        """
        real_data = model_obj["target_data"]

        if method.lower() == "swr":  # sample w/o replacement
            n_real = len(real_data)
            if count <= n_real:
                picks = sample(list(real_data), count)
            else:
                # if user wants more than real_data, after the real unique values are used, we re-sample
                picks = list(real_data)
                overshoot = count - n_real
                picks += sample(list(real_data), overshoot)
            return pd.Series(picks)

        elif method.lower() == "cart":
            # placeholder: random from real
            from numpy.random import default_rng
            rng = default_rng(self.random_state + hash(col) % 999999)
            picks = rng.choice(real_data, size=count, replace=True)
            return pd.Series(picks)

        elif method.lower() == "pmm":
            # placeholder for a real predictive mean matching
            # for now, random
            from numpy.random import default_rng
            rng = default_rng(self.random_state + hash(col) % 999999)
            picks = rng.choice(real_data, size=count, replace=True)
            return pd.Series(picks)

        else:
            # fallback
            from numpy.random import default_rng
            rng = default_rng(self.random_state + hash(col) % 999999)
            picks = rng.choice(real_data, size=count, replace=True)
            return pd.Series(picks)

    # ----------------------------------------------------------------
    # Helper: substitution constraints handling
    # ----------------------------------------------------------------
    def _apply_substitution_constraints(
        self, df: pd.DataFrame, constraints: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        If user provided e.g. constraints = { "N1": ["=", 999], "C2": ["in", [...]] }
        or something more elaborate, we can parse them.
        We'll do a simple approach: for each key, if the op is '=' or '==', we forcibly set that col to the value.
        """
        new_df = df.copy()
        for col, rule in constraints.items():
            # maybe rule is something like ["=", 100], or a tuple, or a list
            # interpret a small handful of possibilities
            if isinstance(rule, list) and len(rule) == 2:
                op = rule[0]
                val = rule[1]
                if op in ["=", "=="]:
                    # direct substitution
                    if col in new_df.columns:
                        new_df[col] = val
        return new_df

plugin = Syn_SeqPlugin