# File: syn_seq.py

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

# method calls
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


def check_rules_violation(
    df: pd.DataFrame,
    target_col: str,
    rules_dict: Dict[str, List[Any]]
) -> pd.Index:
    """
    A function that checks if `target_col` has rules in rules_dict => re-check rows that violate.
      - rules_dict expects something like:
        rules_dict = {
          "target": [
             ("bmi", ">", 0.15),
             ("target", ">", 0)
          ],
          "bp": [
             ("bp", "=", 0)
          ]
        }
    Return the row index that fails the rules for `target_col`.
    """
    if target_col not in rules_dict:
        return pd.Index([])

    sub_rules = rules_dict[target_col]
    mask_valid = pd.Series(True, index=df.index)

    for (col_feat, operator, val) in sub_rules:
        if col_feat not in df.columns:
            # skip if col doesn't exist
            continue

        # Evaluate simple ops
        if operator in ["=", "=="]:
            local_mask = (df[col_feat] == val) | df[col_feat].isna()
        elif operator == ">":
            local_mask = (df[col_feat] > val) | df[col_feat].isna()
        elif operator == ">=":
            local_mask = (df[col_feat] >= val) | df[col_feat].isna()
        elif operator == "<":
            local_mask = (df[col_feat] < val) | df[col_feat].isna()
        elif operator == "<=":
            local_mask = (df[col_feat] <= val) | df[col_feat].isna()
        else:
            # could expand for "!=" or "in", etc.
            local_mask = pd.Series(True, index=df.index)

        mask_valid &= local_mask

    # anything that is NOT valid => violation
    violation_idx = df.index[~mask_valid]
    return violation_idx


class Syn_Seq:
    """
    A sequential-synthesis aggregator for the syn_seq plugin.

      - .fit(...) => column-by-column model building
      - .generate(...) => col-by-col generation
      - if rules => attempt re-generation for rows that violate them
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

        self._first_col = None  # store data for the first col

    def fit(self, loader: Any, *args, **kwargs) -> "Syn_Seq":
        """
        loader => data + info. We read syn_order, method, variable_selection, etc.
        """
        info = loader.info()
        training_data = loader.dataframe().copy()
        if training_data.empty:
            raise ValueError("No data after encoding => cannot train on empty DataFrame.")

        self._syn_order = info["syn_order"]
        self._method_map = info["method"]
        self._varsel = info["variable_selection"]

        print("[INFO] model fitting")
        print("Fitting order =>", self._syn_order)

        for i, col in enumerate(self._syn_order):
            if i == 0:
                # first col => random or swr. We'll just do "swr"
                chosen_m = "swr"
                self._first_col = training_data[col].values.copy()
                print(f"Fitting first col '{col}' with method='{chosen_m}' ... Done!")
                continue

            chosen_m = self._method_map[col]
            preds_list = self._varsel[col]
            y = training_data[col].values
            X = training_data[preds_list].values

            print(f"Fitting '{col}' with method='{chosen_m}' ... ", end="", flush=True)
            self._col_models[col] = self._fit_col(col, y, X, chosen_m)
            print("Done!")

        self._model_trained = True
        return self

    def _fit_col(self, colname: str, y: np.ndarray, X: np.ndarray, method_name: str) -> Dict[str, Any]:
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
        max_iter_rules: int = 10
    ) -> pd.DataFrame:
        """
        Generate 'nrows' new samples, col-by-col. If 'rules' => re-generate violating rows up to max_iter_rules times.
        """
        if not self._model_trained:
            raise RuntimeError("Must fit(...) aggregator before generate().")

        gen_df = pd.DataFrame(index=range(nrows))

        # The first column is just sampling from the original distribution
        first_col_name = self._syn_order[0]
        gen_df[first_col_name] = np.random.choice(self._first_col, size=nrows, replace=True)

        # for the other columns
        for col in self._syn_order[1:]:
            chosen_m = self._method_map[col]
            preds_list = self._varsel[col]

            Xsyn = gen_df[preds_list].values
            ysyn = self._generate_col(col, chosen_m, Xsyn)
            gen_df[col] = ysyn

            if rules and col in rules:
                # re-check violation
                tries = 0
                while True:
                    viol_idx = check_rules_violation(gen_df, col, rules)
                    if len(viol_idx) == 0:
                        break
                    if tries >= max_iter_rules:
                        # set them to NaN or keep them as is
                        gen_df.loc[viol_idx, col] = np.nan
                        print(f"[WARN] {col}: could not satisfy rules after {max_iter_rules} tries => set them to NaN.")
                        break
                    # re-generate only for those violation rows
                    Xsub = gen_df.loc[viol_idx, preds_list].values
                    ysub = self._generate_col(col, chosen_m, Xsub)
                    gen_df.loc[viol_idx, col] = ysub
                    tries += 1

            print(f"Generating '{col}' ... Done!")

        return gen_df

    def _generate_col(self, colname: str, method_name: str, Xsyn: np.ndarray) -> np.ndarray:
        gen_func = METHOD_MAP[method_name][1]
        fitted_model = self._col_models[colname]["fitted_model"]
        return gen_func(fitted_model, Xsyn)
