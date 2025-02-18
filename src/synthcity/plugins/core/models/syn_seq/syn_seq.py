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

# ------------------------------------------------------------------
# Map method names to (training function, generation function)
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Syn_Seq: Column-by-column aggregator for sequential synthesis.
# ------------------------------------------------------------------
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
            strict: (Unused now; rule-checking is handled later.)
            sampling_patience: (Unused now.)
        """
        self.random_state = random_state
        self.strict = strict
        self.sampling_patience = sampling_patience
        self.special_values = Dict[str, List[Any]] = {}
        self._model_trained = False
        self._syn_order: List[str] = []
        self._method_map: Dict[str, str] = {}
        self._varsel: Dict[str, List[str]] = {}
        self._col_models: Dict[str, Dict[str, Any]] = {}

        # Store the real distribution for the first column and columns with special values.
        self._stored_col_data:  = None

    def fit_col(self, loader: DataLoader, *args: Any, **kwargs: Any) -> "Syn_Seq":
        """
        Fit column-by-column using metadata from the loader.
        1) Retrieve info (syn_order, method, variable_selection).
        2) For columns ending with "_cat", force aggregator "cart".
        3) For the first column, store its real distribution.
        4) For each subsequent column, train its aggregator using preceding columns.
        """
        info_dict = loader.info()
        training_data = loader.dataframe().copy()
        if training_data.empty:
            raise ValueError("No data => cannot fit Syn_Seq aggregator")

        self._syn_order = info_dict.get("syn_order", list(training_data.columns))
        self._method_map = info_dict.get("method", {})
        self.special_values = info_dict.get("special_vales", {})
        self._varsel = info_dict.get("variable_selection", {})

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

        # (3) Store the real distribution from the first column.
        first_col = self._syn_order[0]
        self._stored_col_data = training_data[first_col].dropna().values
        print(f"Fitting '{first_col}' => stored distribution from real data. Done.")

        # (4) For each subsequent column, train its aggregator.
        np.random.seed(self.random_state)
        for i, col in enumerate(self._syn_order[1:], start=1):
            method_name = self._method_map.get(col, "cart")
            preds_list = self._varsel.get(col, self._syn_order[:i])
            y = training_data[col].values
            X = training_data[preds_list].values

            mask = (~pd.isna(y))
            X_ = X[mask]
            y_ = y[mask]

            print(f"Fitting '{col}' with '{method_name}' ... ", end="", flush=True)
            self._col_models[col] = self._fit_single_col(method_name, X_, y_)
            print("Done!")
        self._model_trained = True
        return self

    def _fit_single_col(self, method_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        fit_func, _ = METHOD_MAP[method_name]
        try:
            model = fit_func(y, X, random_state=self.random_state)
        except:
        # Need the logic here which if generating the data throws an error due to issues like highly imbalanced data so previous variable has only one value, etc, we need to use the stored data to sample the values of that variable.
        # This only occurrs for columns with special values so I tried to store the data of columns of special values.
        # This means we need to add the fallback here in case of model fitting issue

        return {"name": method_name, "fitted_model": model}

    def generate_col(self, nrows: int) -> pd.DataFrame:
        """
        Generate `nrows` rows, column by column.
        (No rule checking is performed here.)
        """
        if not self._model_trained:
            raise RuntimeError("Syn_Seq aggregator not yet fitted")
        if nrows <= 0:
            return pd.DataFrame(columns=self._syn_order)
        
        gen_df = pd.DataFrame({col: [np.nan] * nrows for col in self._syn_order})
        
        # (1) Generate the first column.
        first_col = self._syn_order[0]
        if self._stored_col_data is not None and len(self._stored_col_data[first_col]) > 0:
            gen_df[first_col] = np.random.choice(self._stored_col_data[first_col], size=nrows, replace=True)
        else:
            gen_df[first_col] = 0
        print(f"Generating '{first_col}' => done.")

        # (2) Generate subsequent columns.
        for col in self._syn_order[1:]:
            method_name = self._method_map.get(col, "cart")
            idx = self._syn_order.index(col)
            preds_list = self._varsel.get(col, self._syn_order[:idx])
            
            Xsyn = gen_df[preds_list].values
            ysyn = self._generate_single_col(method_name, Xsyn, col)
            gen_df[col] = ysyn

            print(f"Generating '{col}' => done.")
        return gen_df

    def _generate_single_col(self, method_name: str, Xsyn: np.ndarray, col: str) -> np.ndarray:

        # Need the logic here which if generating the data throws an error due to issues like highly imbalanced data so previous variable has only one value, etc, we need to use the stored data to sample the values of that variable.
        # This only occurrs for columns with special values so I tried to store the data of columns of special values.
        # This means we need to add the fallback here in case of model fitting issue

        # Following code existed before but I don't understand the usage.
        # if col not in self._col_models:
        #     if self._stored_col_data is not None and len(self._stored_col_data) > 0:
        #         return np.random.choice(self._stored_col_data, size=len(Xsyn), replace=True)
        #     else:
        #         return np.zeros(len(Xsyn))

        fit_info = self._col_models[col]
        _, generate_func = METHOD_MAP[fit_info["name"]]
        return generate_func(fit_info["fitted_model"], Xsyn)
