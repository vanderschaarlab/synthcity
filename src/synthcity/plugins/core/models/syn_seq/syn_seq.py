# File: syn_seq.py

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

# We import the rules class
from synthcity.plugins.core.models.syn_seq.syn_seq_rules import Syn_SeqRules
# We import our method calls from methods/__init__.py
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
    syn_swr, generate_swr
)

# For convenience, map method_name -> (fit_func, generate_func)
METHOD_MAP = {
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
    """
    A sequential-synthesis aggregator that:

      1) For each col in syn_order, pick a method => fit => store in _col_models.
         - The first column can be "swr" or "random" by default if user wants.
      2) generate(...) => for each col in syn_order, generate new y_syn using the fitted model.
      3) if rules => attempt re-generation for those rows that fail the rule constraints, up to 'max_iter'.

      It's recommended that after generating, you pass the result to encoder.inverse_transform(...) to revert cat/dates if needed.
    """

    def __init__(
        self,
        random_state: int = 0,
        strict: bool = True,
        sampling_patience: int = 100,
    ):
        self.random_state = random_state
        self.strict = strict
        self.sampling_patience = sampling_patience
        self._col_models: Dict[str, Dict[str, Any]] = {}
        self._model_trained = False
        self._syn_order: List[str] = []
        self._method_map: Dict[str, str] = {}
        self._varsel: Dict[str, List[str]] = {}

        self._first_col = None  # We'll store the distribution or data for the first col.

    def fit(self, loader: Any, *args, **kwargs) -> "Syn_Seq":
        """
        Args:
            loader: the fitted DataLoader (Syn_SeqDataLoader).
                   We read syn_order, method, variable_selection from loader.info
        """
        info = loader.info()
        training_data = loader.dataframe().copy()
        if training_data.empty:
            raise ValueError("No data after encoding => cannot train on empty DataFrame.")

        syn_order: List[str] = info["syn_order"]
        method_map: Dict[str, str] = info["method"]  # col-> "rf"/"norm"/"cart" ...
        varsel: Dict[str, List[str]] = info["variable_selection"]  # dict

        self._syn_order = syn_order
        self._method_map = method_map
        self._varsel = varsel

        print("[INFO] model fitting")
        print("Fitting order =>", syn_order)

        # Fit logic
        for i, col in enumerate(syn_order):
            if i == 0:
                # We'll treat the first col as "swr" by default
                chosen_m = "swr"
                self._first_col = training_data[col].values.copy()
                print(f"Fitting first col '{col}' with method='{chosen_m}' ... Done (no real model).")
                continue

            chosen_m = method_map[col]
            preds_list = varsel[col]
            if len(preds_list) == 0:
                # e.g. second col might only have the first col as predictor => preds_list = [syn_order[0]], etc.
                pass
            y = training_data[col].values
            X = training_data[preds_list].values

            print(f"Fitting '{col}' with method='{chosen_m}' ... ", end="", flush=True)
            model_dict = self._fit_col(col, y, X, chosen_m)
            self._col_models[col] = model_dict
            print("Done!")

        self._model_trained = True
        return self

    def _fit_col(self, colname: str, y: np.ndarray, X: np.ndarray, method_name: str) -> Dict[str, Any]:
        """
        Fit a single column's model using the relevant method from METHOD_MAP.
        Return a dictionary that stores the fitted model (and anything else).
        """
        fit_func, _ = METHOD_MAP[method_name]
        model = fit_func(y, X, random_state=self.random_state)
        return {
            "name": method_name,
            "fitted_model": model,
        }

    def generate(
        self,
        nrows: int,
        rules: Optional[Dict[str, List[Any]]] = None,
        *args, **kwargs
    ) -> pd.DataFrame:
        """
        Generate 'nrows' new samples, col-by-col. If rules => we re-generate for violations.
        Return a raw DataFrame with the same columns as self._syn_order (including the _cat columns).
        """
        if not self._model_trained:
            raise RuntimeError("Must fit(...) aggregator before .generate().")

        # We'll create an empty DataFrame for the results
        gen_df = pd.DataFrame(index=range(nrows))

        # Set the first column
        first_col_name = self._syn_order[0]
        gen_df[first_col_name] = np.random.choice(self._first_col, size=nrows, replace=True)

        # Prepare rules
        rules_obj = None
        if rules is not None:
            rules_obj = Syn_SeqRules(chained_rules=rules, max_iter=10)

        # For each subsequent column
        for col in self._syn_order[1:]:
            chosen_m = self._method_map[col]
            preds_list = self._varsel[col]
            X_syn = gen_df[preds_list].values  # shape (nrows, #predictors)

            # Use fitted model to generate
            y_syn = self._generate_col(col, chosen_m, X_syn)
            gen_df[col] = y_syn

            # If rules => re-check rows that violate
            if rules_obj is not None and col in rules_obj.chained_rules:
                iter_count = 0
                while True:
                    viol_idx = rules_obj.check_violations(gen_df, col)
                    if len(viol_idx) == 0:
                        break  # no more violations => done
                    if iter_count >= rules_obj.max_iter:
                        # we keep them as is => or set them to NaN
                        gen_df.loc[viol_idx, col] = np.nan
                        print(f"[WARN] Could not fully satisfy rules for '{col}' after {rules_obj.max_iter} tries. Setting them to NaN.")
                        break
                    # re-generate only for those violation rows
                    X_syn_sub = gen_df.loc[viol_idx, preds_list].values
                    y_syn_sub = self._generate_col(col, chosen_m, X_syn_sub)
                    gen_df.loc[viol_idx, col] = y_syn_sub
                    iter_count += 1

            print(f"Generating '{col}' ... Done!")

        return gen_df

    def _generate_col(self, colname: str, method_name: str, X_syn: np.ndarray) -> np.ndarray:
        """
        Call the 'generate_<method>' function to synthesize the target col from the model.
        """
        model_dict = self._col_models[colname]
        gen_func = METHOD_MAP[method_name][1]  # second element: generate_func
        # fitted_model is what we stored in model_dict["fitted_model"]
        y_syn = gen_func(model_dict["fitted_model"], X_syn)
        return y_syn
