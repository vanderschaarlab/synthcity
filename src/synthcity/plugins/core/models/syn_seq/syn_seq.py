"""
Revised syn_seq.py

This module implements a sequential synthesizer (Syn_Seq) that fits each column 
of the data (following an ordering provided in syn_order) one by one.
It supports columns with special values. For such columns, during fitting the 
model only sees rows with numeric (non‐special) values. At generation time, a 
synthetic _cat indicator is sampled from the training distribution (which includes 
the numeric marker and any special values) so that:
  - If the synthetic _cat value equals the numeric marker (DEFAULT_NUMERIC_MARKER), 
    the fitted model is used to generate a numeric value.
  - Otherwise, the generated base column value is overridden by the special value.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import warnings

# Import column-fitting and generation functions for various methods.
from synthcity.plugins.core.models.syn_seq.methods import (
    syn_cart, generate_cart,
    syn_ctree, generate_ctree,
    syn_logreg, generate_logreg,
    syn_norm, generate_norm,
    syn_pmm, generate_pmm,
    syn_polyreg, generate_polyreg,
    syn_rf, generate_rf,
    syn_lognorm, generate_lognorm,
    syn_random, generate_random,
    syn_swr, generate_swr,
)

# Map method names to (fitting function, generation function)
METHOD_MAP: Dict[str, Tuple[Any, Any]] = {
    "cart": (syn_cart, generate_cart),
    "ctree": (syn_ctree, generate_ctree),
    "logreg": (syn_logreg, generate_logreg),
    "norm": (syn_norm, generate_norm),
    "pmm": (syn_pmm, generate_pmm),
    "polyreg": (syn_polyreg, generate_polyreg),
    "rf": (syn_rf, generate_rf),
    "lognorm": (syn_lognorm, generate_lognorm),
    "random": (syn_random, generate_random),
    "swr": (syn_swr, generate_swr),
}

NUMERIC_MARKER = -777777777
MISSING_MARKER = -999999999

class Syn_Seq:
    def __init__(
        self,
        random_state: int = 0,
        sampling_patience: int = 100
    ):
        """
        Args:
            random_state: Random seed.
            sampling_patience: Maximum number of attempts in generation.
        """
        self.random_state = random_state
        self.sampling_patience = sampling_patience
        self.cat_distributions: Dict[str, Dict[Any, float]] = {}
        self._model_trained = False
        self._syn_order: List[str] = []     
        self._method_map: Dict[str, str] = {}  
        self._varsel: Dict[str, List[str]] = {} 
        self._col_models: Dict[str, Dict[str, Any]] = {}  
        self._first_col_values: Dict[str, np.ndarray] = {}

    def fit_col(self, loader: Any, label_encoder: Any, *args: Any, **kwargs: Any) -> "Syn_Seq":
        """
        Fit each column sequentially using metadata from the loader.
        For each _cat column in the training data, record its full distribution (casting values to int)
        and record the list of special values. Then, for base columns with special values,
        filter out rows whose value is special (so that the model sees only numeric values).
        """
        info_dict = loader.info()
        training_data = loader.dataframe().copy()
        if training_data.empty:
            raise ValueError("No data => cannot fit Syn_Seq aggregator")
        # Set synthesis order, method mapping, and variable selection from loader info.
        self._syn_order = info_dict.get("syn_order", list(training_data.columns))
        self._method_map = info_dict.get("method", {})
        self._varsel = info_dict.get("variable_selection", {})
        # --- Build the _cat column distributions ---
        for col in training_data.columns:
            if col.endswith("_cat"):
                base_col = col[:-4]
                unique_vals = training_data[col].dropna().unique().tolist()
                self.cat_distributions[col] = # calculate the proportion of each unique value

        # For auto-injected _cat columns, force method "cart" and mirror variable selection.
        for col in self._syn_order:
            if col.endswith("_cat"):
                self._method_map[col] = "cart"
                base_col = col[:-4]
                if base_col in self._varsel:
                    self._varsel[col] = self._varsel[base_col]
                else:
                    idx = self._syn_order.index(col)
                    self._varsel[col] = self._syn_order[:idx]

        print("[INFO] Syn_Seq aggregator: fitting columns...")

        # For the first column, store its observed (non-null) distribution.
        first_col = self._syn_order[0]
        self._first_col_values[first_col] = training_data[first_col].dropna().values

        print(f"Fitting '{first_col}' => stored values from real data. Done.")

        # Fit a model for each subsequent column.
        np.random.seed(self.random_state)
        for i, col in enumerate(self._syn_order[1:], start=1):
            method_name = self._method_map.get(col, "cart")
            preds_list = self._varsel.get(col, self._syn_order[:i])
            y = training_data[col].values
            X = training_data[preds_list].values
            cat_col = col + "_cat"
            if cat_col in preds_list:
                numeric_label = index(label_encoder[cat_col].classes_ == NUMERIC_MARKER)
                mask = (y == numeric_label)
                y = y[mask]
                X = X[mask]

            print(f"Fitting '{col}' with '{method_name}' ... ", end="", flush=True)
            try:
                self._col_models[col] = self._fit_single_col(method_name, X, y)
            except Exception as e:
                print(f"Error fitting column {col} with {method_name}: {e}.", end=" ")
                self._col_models[col] = None
                    
            print("Done!")
        self._model_trained = True
        return self

    def _fit_single_col(self, method_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Fit a single column using the specified method.
        """
        fit_func, _ = METHOD_MAP[method_name]
        model = fit_func(y, X, random_state=self.random_state)
        return {"name": method_name, "fitted_model": model}

    def generate_col(self, label_encoder, count: int) -> pd.DataFrame:
        """
        Generate `count` rows sequentially.
        
        For each base column that has a corresponding _cat distribution, we first generate the
        base column using its fitted model. Then we sample a synthetic _cat indicator (using the 
        saved full distribution). For rows where the synthetic indicator is not equal to the numeric
        marker, we override the generated base column value with the special value.
        """
        if not self._model_trained:
            raise RuntimeError("Syn_Seq aggregator not yet fitted")
        if count <= 0:
            return pd.DataFrame(columns=self._syn_order)
        
        # Initialize a DataFrame with NaN values for all columns.
        gen_df = pd.DataFrame({col: [np.nan] * count for col in self._syn_order})
        
        # (1) Generate the first column using its stored distribution.
        first_col = self._syn_order[0]
        if first_col in self._first_col_values and len(self._first_col_values[first_col]) > 0:
            gen_df[first_col] = np.random.choice(
                self._first_col_values[first_col], size=count, replace=True
            )
        else:
            print("Error generating first column, storing zeroes.", end=" ")
            gen_df[first_col] = 0
        print(f"Generating '{first_col}' => done.")
        
        # (2) For each subsequent column, generate synthetic values.
        for col in self._syn_order[1:]:
            method_name = self._method_map.get(col, "cart")
            preds_list = self._varsel.get(col, self._syn_order[:self._syn_order.index(col)])
            cat_col = col + "_cat"
            if cat_col in preds_list:
                Xsyn_numeric = Xsyn[Xsyn[cat_col] == index(label_encoder[cat_col].classes_ == NUMERIC_MARKER)].values
                Xsyn_categorical = Xsyn[~(Xsyn[cat_col] == index(label_encoder[cat_col].classes_ == NUMERIC_MARKER))]

                ysyn_numeric = self._generate_single_col(method_name, Xsyn_numeric, col)
                ysyn_categorical = Xsyn_categorical[cat_col]

                # Need the logic to concatenate them and finalize the gen_df

            # If _cat column is not in pred list, do the normal generation
            Xsyn = gen_df[preds_list].values
            ysyn = self._generate_single_col(method_name, Xsyn, col)
            gen_df[col] = ysyn
            print(f"Generating '{col}' => done.")
            
            if col in self.cat_distributions:
                # This is to check if _cat column doesn't have any 'numeric' flag. If that happens, it means model failed to fit the imbalanced data so we fallback.
                numeric_label = idx(label_encoder[col].classes_ == NUMERIC_MARKER)
                if numeric_label not in gen_df[col]:
                    print(f"{col} does not contain indicator for numeric values for its base column. model might have failed to fit the data due to highly skewed distribution. Using empirical distribution for generation.")
                    gen_df[col] = np.nan
                    cat_dist = self.cat_distributions[col]
                    #Here, generate the gen_df[col] according to its empirical distribution.
        
        return gen_df

    def _generate_single_col(self, method_name: str, Xsyn: np.ndarray, col: str) -> np.ndarray:
        """
        Generate synthetic values for a single column using the fitted model.
        If no model is available for the column, a RuntimeError is raised.
        """
        if col not in self._col_models or self._col_models[col] is None:
            raise RuntimeError(f"No model available for column {col}.")
        fit_info = self._col_models[col]
        _, generate_func = METHOD_MAP[fit_info["name"]]
        return generate_func(fit_info["fitted_model"], Xsyn)
