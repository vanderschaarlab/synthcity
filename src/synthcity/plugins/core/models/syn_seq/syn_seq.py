from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from synthcity.logger import info, warning
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.models.syn_seq.syn_seq_encoder import Syn_SeqEncoder


# Import the column-fitting and column-generating functions.
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

# Map method names to (training function, generation function)
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


class Syn_Seq:
    def __init__(
        self,
        random_state: int = 0,
        strict: bool = True,
        sampling_patience: int = 100
    ):
        """
        Args:
            random_state: Random seed.
            strict: (Unused now; rule‐checking is handled later.)
            sampling_patience: (Unused now.)
        """
        self.random_state = random_state
        self.strict = strict
        self.sampling_patience = sampling_patience
        self.special_values: Dict[str, List[Any]] = {}  # mapping: col -> list of special values
        self._model_trained = False
        self._syn_order: List[str] = []
        self._method_map: Dict[str, str] = {}
        self._varsel: Dict[str, List[str]] = {}
        self._col_models: Dict[str, Dict[str, Any]] = {}

        # Store the real distribution for the first column and for columns with special values.
        self._stored_col_data: Dict[str, np.ndarray] = {}

    def fit_col(self, loader: Any, *args: Any, **kwargs: Any) -> "Syn_Seq":
        """
        Fit column‐by‐column using metadata from the loader.
        """
        info_dict = loader.info()
        training_data = loader.dataframe().copy()
        if training_data.empty:
            raise ValueError("No data => cannot fit Syn_Seq aggregator")

        # Set syn_order, method mapping, variable selection, and special values.
        self._syn_order = info_dict.get("syn_order", list(training_data.columns))
        self._method_map = info_dict.get("method", {})
        self.special_values = info_dict.get("special_values", {})
        self._varsel = info_dict.get("variable_selection", {})

        # For auto-injected _cat columns, force aggregator "cart" and mirror variable selection.
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

        # (3) Store the real distribution from the first column.
        first_col = self._syn_order[0]
        self._stored_col_data[first_col] = training_data[first_col].dropna().values

        # For columns with special values, store all non-null values that are NOT special.
        for col, specials in self.special_values.items():
            # Filter rows where the column's value is not in specials.
            filtered = training_data[~training_data[col].isin(specials)]
            # Drop any NaNs and store the underlying values.
            self._stored_col_data[col] = filtered[col].dropna().values

        print(f"Fitting '{first_col}' => stored distribution from real data. Done.")

        # (4) For each subsequent column, train its aggregator.
        np.random.seed(self.random_state)
        for i, col in enumerate(self._syn_order[1:], start=1):
            method_name = self._method_map.get(col, "cart")
            preds_list = self._varsel.get(col, self._syn_order[:i])
            y = training_data[col].values
            X = training_data[preds_list].values
            mask = (~pd.isna(y))
            # If the column has special values, drop rows where y is one of those special values.
            if col in self.special_values:
                specials = self.special_values[col]
                mask = mask & (~np.isin(y, specials))
            X_ = X[mask]
            y_ = y[mask]
            print(f"Fitting '{col}' with '{method_name}' ... ", end="", flush=True)
            try:
                self._col_models[col] = self._fit_single_col(method_name, X_, y_)
            except Exception as e:
                print(f"Error fitting column {col}: {e}. Falling back to swr.", end=" ")
                try:
                    self._col_models[col] = self._fit_single_col("swr", X, y)
                except Exception as e2:
                    print(f"Fallback swr also failed for {col}: {e2}. Storing None.", end=" ")
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

    def generate_col(self, count: int) -> pd.DataFrame:
        """
        Generate `count` rows, column-by-column.
        """
        if not self._model_trained:
            raise RuntimeError("Syn_Seq aggregator not yet fitted")
        if count <= 0:
            return pd.DataFrame(columns=self._syn_order)
        
        # Initialize a DataFrame with NaN values.
        gen_df = pd.DataFrame({col: [np.nan] * count for col in self._syn_order})
        
        # (1) Generate the first column using the stored real distribution.
        first_col = self._syn_order[0]
        if self._stored_col_data.get(first_col) is not None and len(self._stored_col_data[first_col]) > 0:
            gen_df[first_col] = np.random.choice(self._stored_col_data[first_col], size=count, replace=True)
        else:
            gen_df[first_col] = 0
        print(f"Generating '{first_col}' => done.")
        
        # (2) Generate subsequent columns.
        for col in self._syn_order[1:]:
            method_name = self._method_map.get(col, "cart")
            idx = self._syn_order.index(col)
            preds_list = self._varsel.get(col, self._syn_order[:idx])

            if col in self.special_values:
                Xsyn_num = gen_df[gen_df["f{col}_col"] == "NUMERIC"]
                ysyn_num =  self._generate_single_col(method_name, Xsyn_num, col)
                Xsyn_special = gen_df[~(gen_df["f{col}_col"] == "NUMERIC")]
                ysyn_special =  self._generate_single_col(method_name, Xsyn_special, col)
                Xsyn = pd.concat(Xsyn_num, Xsyn_special)
                ysyn = pd.concat(ysyn_num, ysyn_special)
            else:
                Xsyn = gen_df[preds_list].values
                ysyn = self._generate_single_col(method_name, Xsyn, col)
            gen_df[col] = ysyn
            print(f"Generating '{col}' => done.")
        return gen_df

    def _generate_single_col(self, method_name: str, Xsyn: np.ndarray, col: str) -> np.ndarray:
        """
        Generate synthetic values for a single column using the fitted model.
        """
        if col not in self._col_models or self._col_models[col] is None:
            if col in self._stored_col_data and len(self._stored_col_data[col]) > 0:
                return np.random.choice(self._stored_col_data[col], size=len(Xsyn), replace=True)
            else:
                return np.zeros(len(Xsyn))
        
        fit_info = self._col_models[col]
        _, generate_func = METHOD_MAP[fit_info["name"]]
        return generate_func(fit_info["fitted_model"], Xsyn)
