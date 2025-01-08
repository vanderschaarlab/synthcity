# File: plugins/syn_seq/plugin_syn_seq.py

from typing import Any, Dict, Optional

import pandas as pd

# synthcity absolute
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader
from synthcity.plugins.core.schema import Schema

# local references
from .syn_seq_dataloader import Syn_SeqDataLoader
from .syn_seq_encoder import Syn_SeqEncoder


class Syn_SeqPlugin(Plugin):
    """
    A plugin for sequential (column-by-column) synthetic data generation,
    all contained in this single class (no separate 'Synthesizer').

    Workflow:

        1. Provide a Syn_SeqDataLoader with your DF, plus syn_order, special values, etc.
        2. Call .fit(dataloader, method=..., variable_selection=...).
           - We encode the data,
           - Then for each column, train a model (e.g., SWR, CART, PMM, etc.)
             using the user-defined or default methods & variable selection.
        3. Call .generate(count=...), which uses the stored models to
           create new rows column-by-column, optionally decode them,
           and return a brand-new Syn_SeqDataLoader.
    """

    # ------------------------------------------------
    # Plugin "metadata" for synthcity
    # ------------------------------------------------
    @staticmethod
    def name() -> str:
        return "syn_seq"

    @staticmethod
    def type() -> str:
        # Typically "generic," or you could define a new category.
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> list:
        """
        If you want to expose tuneable hyperparameters for AutoML,
        define them here (e.g., distributions for 'n_estimators', etc.).
        For now, returning [] means "no hyperparams."
        """
        return []

    # ------------------------------------------------
    # Initialization
    # ------------------------------------------------
    def __init__(
        self,
        random_state: int = 0,
        # You can add other plugin-level arguments if needed:
        default_first_method: str = "SWR",
        default_other_method: str = "CART",
        **kwargs: Any
    ):
        """
        Args:
            random_state: for reproducibility.
            default_first_method: if user doesn't override, the first column uses SWR.
            default_other_method: if user doesn't override, other columns use CART.
            **kwargs: optional plugin or parent-Plugin arguments (strict, workspace, etc.)
        """
        super().__init__(random_state=random_state, **kwargs)

        # Just store some defaults or extra config
        self.default_first_method = default_first_method
        self.default_other_method = default_other_method

        # For tracking the fitted “models” per column
        self._column_models: Dict[str, Any] = {}

        # For tracking the "method" mapping and "variable_selection"
        self.method_map: Dict[str, str] = {}
        self.variable_selection: Dict[str, Any] = {}

        self._encoders: Dict[str, Any] = {}
        self._model_trained = False

    # ------------------------------------------------
    # Public .fit()
    # ------------------------------------------------
    def fit(
        self,
        dataloader: Syn_SeqDataLoader,
        method: Optional[Dict[str, str]] = None,
        variable_selection: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> "Syn_SeqPlugin":
        """
        Main entrypoint to train/fix column-by-column "models."

        Args:
            dataloader: a Syn_SeqDataLoader with your real dataset.
            method: dict specifying the method for each column, e.g. {"colA":"pmm"}.
                    If a column is not in `method`, fallback to defaults.
            variable_selection: dict controlling what predictor columns are used
                                for each target column.
                                e.g. {"colA":[0,1,1,0], "colB":[1,0,1,1], ...}
                                Or any structure you prefer.

        Steps:
            1) Encode the data if no external encoders provided.
            2) Store user method map & variable selection.
            3) Call parent plugin's .fit() => triggers _fit().
        """
        # Quick type-check
        if not isinstance(dataloader, Syn_SeqDataLoader):
            raise TypeError("Syn_SeqPlugin requires a Syn_SeqDataLoader.")

        # Encode if needed
        encoded_loader, encoders = dataloader.encode()
        df_encoded = encoded_loader.dataframe()
        if df_encoded.empty:
            raise ValueError("No data to train on in Syn_SeqPlugin.")

        self._encoders = encoders  # store for usage in .generate
        self.method_map = method or {}
        self.variable_selection = variable_selection or {}

        # Now let the base Plugin do the rest (it calls _fit internally).
        super().fit(encoded_loader, *args, **kwargs)
        return self

    # ------------------------------------------------
    # The parent's .fit() calls ._fit() internally
    # ------------------------------------------------
    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "Syn_SeqPlugin":
        """
        The internal training routine.
        We'll do column-by-column approach here:
          1) For each column in X, decide the method.
          2) Build a simple model or approach (SWR, CART, etc.).
          3) Save to self._column_models[col].
        """
        df = X.dataframe()
        col_list = list(df.columns)

        for i, col in enumerate(col_list):
            # Pick method: user override or fallback
            chosen_method = self.method_map.get(col, None)
            if not chosen_method:
                chosen_method = (
                    self.default_first_method if i == 0 else self.default_other_method
                )

            # Decide predictor columns from variable_selection
            # e.g. you might store "col": ["colA","colB"] or a mask, etc.
            predictor_cols = self._get_predictors(col, col_list)

            # Implement or call an internal function to fit a model
            model_obj = self._train_column_model(
                df,
                target_col=col,
                method=chosen_method,
                predictor_cols=predictor_cols,
            )

            # Store it
            self._column_models[col] = {
                "method": chosen_method,
                "predictors": predictor_cols,
                "model": model_obj,
            }

        self._model_trained = True
        return self

    # ------------------------------------------------
    # Public .generate()
    # ------------------------------------------------
    def generate(
        self,
        count: int = 10,
        *args: Any,
        **kwargs: Any
    ) -> Syn_SeqDataLoader:
        """
        Synthesize new samples.

        Steps:
            1) Call parent's generate => triggers _generate().
            2) If we have a stored "syn_seq_encoder," decode columns.
            3) Return as a new Syn_SeqDataLoader.
        """
        if not self._model_trained:
            raise RuntimeError("Syn_SeqPlugin: please call fit() before generate().")

        # Let the base plugin handle constraints + calling _generate
        gen_data: DataLoader = super().generate(count=count, *args, **kwargs)

        # Optionally decode if we have an encoder
        syn_df = gen_data.dataframe()
        if "syn_seq_encoder" in self._encoders:
            encoder = self._encoders["syn_seq_encoder"]
            syn_df = encoder.inverse_transform(syn_df)

        # Wrap in a new Syn_SeqDataLoader
        syn_loader = Syn_SeqDataLoader(
            data=syn_df,
            syn_order=list(syn_df.columns),  # or preserve original order if you prefer
        )
        return syn_loader

    # ------------------------------------------------
    # The parent's .generate() calls ._generate()
    # ------------------------------------------------
    def _generate(
        self,
        count: int,
        syn_schema: Schema,
        **kwargs: Any,
    ) -> DataLoader:
        """
        The internal generation routine:
          - We'll build up a new DF row-by-row or column-by-column.
          - For each column, call the stored model to produce synthetic data.
        """
        if not self._model_trained:
            raise RuntimeError("Must fit() the plugin before generating data.")

        # If we have the same column order used during training, use that.
        col_list = list(self._column_models.keys())

        # Create an empty df with `count` rows
        syn_df = pd.DataFrame(index=range(count))

        # Fill column by column
        for i, col in enumerate(col_list):
            info = self._column_models[col]
            chosen_method = info["method"]
            predictor_cols = info["predictors"]
            model_obj = info["model"]

            # We'll call an internal function to do the actual sampling
            col_values = self._generate_for_column(
                count,
                col,
                chosen_method,
                model_obj,
                predictor_cols,
                syn_df,
            )
            syn_df[col] = col_values

        # Return a DataLoader. The base plugin just needs *some* DataLoader.
        # A GenericDataLoader is enough for now. Or you could wrap a Syn_SeqDataLoader if desired.
        return GenericDataLoader(syn_df)

    # ------------------------------------------------
    # Helper: get predictor columns for a given col
    # ------------------------------------------------
    def _get_predictors(self, col: str, col_list: list) -> list:
        """
        For example, if variable_selection is like:
            {
              "colA": ["col1","col2"],
              "colB": ...
            }
        or a mask, or anything the user provided.
        If not found, fallback to some default logic (like "all previous columns").
        """
        if col in self.variable_selection:
            # e.g. variable_selection[col] = ["colX", "colY"]
            preds = self.variable_selection[col]
            if isinstance(preds, list):
                return preds
            # Or handle other data structure
        else:
            # fallback: maybe all columns before `col` in col_list?
            idx = col_list.index(col)
            preds = col_list[:idx]
        return preds

    # ------------------------------------------------
    # Helper: train a "model" for one column
    # ------------------------------------------------
    def _train_column_model(
        self,
        df: pd.DataFrame,
        target_col: str,
        method: str,
        predictor_cols: list,
    ) -> Any:
        """
        This function trains the actual column-level model. E.g.:
          - If method == "SWR", we store the observed distribution or do sampling w/o replacement
          - If method == "CART", we train a decision tree regressor or classifier
          - etc.
        For this minimal example, we won't do anything complex.
        We just store the column's data for reference. :-)
        """
        # We'll store a dict with the "training data" so we can sample later.
        # Replace with your real method code (like scikit-learn, etc.)
        train_info = {
            "method": method,
            "predictor_cols": predictor_cols,
            "target_data": df[target_col].values,
        }
        return train_info

    # ------------------------------------------------
    # Helper: generate data for one column
    # ------------------------------------------------
    def _generate_for_column(
        self,
        count: int,
        col: str,
        method: str,
        model_obj: Any,
        predictor_cols: list,
        partial_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Given the chosen method & the partial DF so far (which may contain
        already-synthesized columns), produce `count` new values for col.

        For example:
          - SWR => sample randomly from the real values (w/o replacement).
          - CART => run a trained decision tree on partial_df's predictor columns.
          - PMM => do predictive mean matching, etc.

        This minimal example just does random sampling from the real column data we stored.
        """
        real_data = model_obj["target_data"]
        if method == "SWR":
            # naive "sample w/o replacement" if count <= len(real_data)
            # else we do a repeated sample just for demonstration
            from random import sample
            n_real = len(real_data)
            if count <= n_real:
                return pd.Series(sample(list(real_data), count))
            else:
                # sample all then repeat
                picks = list(real_data)
                overshoot = count - n_real
                picks += sample(list(real_data), overshoot)
                return pd.Series(picks)
        elif method == "CART":
            # placeholder: real CART logic would predict partial_df[predictor_cols]
            # Here just random from real data as a placeholder
            from numpy.random import choice
            return pd.Series(choice(real_data, size=count, replace=True))
        else:
            # fallback
            from numpy.random import choice
            return pd.Series(choice(real_data, size=count, replace=True))
