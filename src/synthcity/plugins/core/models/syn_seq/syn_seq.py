# File: syn_seq.py

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

# from .methods import ... => You must ensure the below imports point to your actual method files
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
    Checks if `target_col` has rules in rules_dict => returns the row index that fails.
    rules_dict example:
        {
          "target": [
             ("bmi", ">", 0.15),
             ("target", ">", 0)
          ],
          ...
        }
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
            local_mask = pd.Series(True, index=df.index)

        mask_valid &= local_mask

    violation_idx = df.index[~mask_valid]
    return violation_idx


class Syn_Seq:
    """
    A simple sequential-synthesis aggregator:

      - .fit(dataloader) => for each col in order => builds model using chosen method
      - .generate(nrows, rules=...) => col-by-col generation; if rules => re-generate violation rows
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

        self._first_col_name: Optional[str] = None
        self._first_col_data: Optional[np.ndarray] = None

    def fit(self, loader: Any, *args, **kwargs) -> "Syn_Seq":
        """
        loader => data + info (including syn_order, method, variable_selection).
        We'll read from loader.info() the 'syn_order', 'method', 'variable_selection' etc.
        """
        info = loader.info()  # expecting syn_order, col_map, variable_selection, ...
        training_data = loader.dataframe().copy()
        if training_data.empty:
            raise ValueError("No data after encoding => cannot train on empty DataFrame.")

        self._syn_order = info["syn_order"]
        # build method map => col => method
        # store variable_selection => col => [predictors]
        col_map = info.get("col_map", {})
        self._method_map = {}
        for col in self._syn_order:
            m = "cart"
            if col in col_map and "method" in col_map[col]:
                m = col_map[col]["method"]
            self._method_map[col] = m

        self._varsel = info.get("variable_selection", {})

        print("[INFO] model fitting")

        # handle first column data
        self._first_col_name = self._syn_order[0]
        self._first_col_data = training_data[self._first_col_name].values.copy()

        for i, col in enumerate(self._syn_order):
            chosen_m = self._method_map[col]
            if i == 0:
                # no actual model for the first col => we rely on sampling from original
                print(f"Fitting first col '{col}' with method='{chosen_m}' ... Done!")
                continue

            # build model
            preds_list = self._varsel.get(col, self._syn_order[:i])
            y = training_data[col].values
            Xpred = training_data[preds_list].values if preds_list else None

            print(f"Fitting '{col}' with method='{chosen_m}' ... ", end="", flush=True)
            self._col_models[col] = self._fit_col(col, y, Xpred, chosen_m)
            print("Done!")

        self._model_trained = True
        return self

    def _fit_col(self, colname: str, y: np.ndarray, X: Optional[np.ndarray], method_name: str) -> Dict[str, Any]:
        if method_name not in METHOD_MAP:
            # fallback
            method_name = "cart"
        fit_func, _ = METHOD_MAP[method_name]
        if X is None or X.shape[1] == 0:
            # For a column with zero predictors, we can pass X of shape (n,0) or do fallback
            X = np.zeros((len(y), 0))
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
        if not self._model_trained:
            raise RuntimeError("Must fit(...) aggregator before generate().")

        gen_df = pd.DataFrame(index=range(nrows))

        # 1) first col => sample from original distribution
        first_col = self._first_col_name
        if not isinstance(self._first_col_data, np.ndarray):
            raise RuntimeError("No first col data stored.")
        gen_df[first_col] = np.random.choice(self._first_col_data, size=nrows, replace=True)

        # 2) for each subsequent col
        for i, col in enumerate(self._syn_order):
            if i == 0:
                continue
            chosen_m = self._method_map[col]
            preds_list = self._varsel.get(col, self._syn_order[:i])
            Xsyn = gen_df[preds_list].values if preds_list else np.zeros((nrows, 0))

            ysyn = self._generate_col(col, chosen_m, Xsyn)
            gen_df[col] = ysyn

            # re-check rules
            if rules and col in rules:
                tries = 0
                while True:
                    viol_idx = check_rules_violation(gen_df, col, rules)
                    if len(viol_idx) == 0:
                        break
                    if tries >= max_iter_rules:
                        # set them to NaN
                        gen_df.loc[viol_idx, col] = np.nan
                        print(f"[WARN] {col}: could not satisfy rules after {max_iter_rules} tries => set them to NaN.")
                        break
                    # re-generate only for violation rows
                    Xsub = gen_df.loc[viol_idx, preds_list].values
                    ysub = self._generate_col(col, chosen_m, Xsub)
                    gen_df.loc[viol_idx, col] = ysub
                    tries += 1

            print(f"Generating '{col}' ... Done!")

        return gen_df

    def _generate_col(self, colname: str, method_name: str, Xsyn: np.ndarray) -> np.ndarray:
        gen_func = METHOD_MAP[method_name][1]
        model_info = self._col_models[colname]
        fitted_model = model_info["fitted_model"]
        return gen_func(fitted_model, Xsyn)
