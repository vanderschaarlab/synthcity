# File: syn_seq.py

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Import the syn_* and generate_* methods from submodules
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
    Given a dictionary like:
       rules = {
         "target": [("bmi", ">", 0.15), ("target", ">", 0)],
         "bp": [("bp", "=", 0)]
       }
    For the target_col in question, find any row indices that fail the rule(s).
    The row is considered valid if it meets *all* of that col's rules or is NA in that column.
    """
    if target_col not in rules_dict:
        return pd.Index([])

    sub_rules = rules_dict[target_col]
    mask_valid = pd.Series(True, index=df.index)

    for (col_feat, operator, val) in sub_rules:
        if col_feat not in df.columns:
            # skip if col doesn't exist
            continue

        # Evaluate ops
        if operator in ["=", "=="]:
            cond = (df[col_feat] == val) | df[col_feat].isna()
        elif operator == ">":
            cond = (df[col_feat] > val) | df[col_feat].isna()
        elif operator == ">=":
            cond = (df[col_feat] >= val) | df[col_feat].isna()
        elif operator == "<":
            cond = (df[col_feat] < val) | df[col_feat].isna()
        elif operator == "<=":
            cond = (df[col_feat] <= val) | df[col_feat].isna()
        else:
            cond = pd.Series(True, index=df.index)  # skip unrecognized op

        mask_valid &= cond

    return df.index[~mask_valid]


class Syn_Seq:
    """
    The aggregator for sequential-synthesis:

      - .fit(loader) : read syn_order, method, varsel, etc from loader.info(), 
        force method="cart" for _cat columns, train each col in order
      - .generate(nrows, rules, ...) : sample first col from real distribution, 
        then generate each subsequent col, re-generating rows that violate user rules.

    The final special-value or numeric restoration is done in the inverse_transform 
    by the syn_seq_encoder after generation. 
    """

    def __init__(
        self,
        random_state: int = 0,
        strict: bool = True,
        sampling_patience: int = 100,
    ):
        """
        Args:
            random_state: seed for reproducibility
            strict: if True, tries to keep re-generating for rule violations
            sampling_patience: max tries for rule-based re-generation
        """
        self.random_state = random_state
        self.strict = strict
        self.sampling_patience = sampling_patience

        self._model_trained = False

        self._syn_order: List[str] = []
        self._method_map: Dict[str, str] = {}
        self._varsel: Dict[str, List[str]] = {}
        self._col_models: Dict[str, Dict[str, Any]] = {}

        # For the first column, we store the real distribution to do "swr" or "random"
        self._first_col_data: Optional[np.ndarray] = None

    def fit(self, loader: Any, *args, **kwargs) -> "Syn_Seq":
        """
        Fit the aggregator from encoded data in 'loader'.
        We:
          1. Read syn_order, method, variable_selection from loader.info().
          2. Force any column ending with '_cat' => method='cart'
             Also copy the base col's varsel if base col in varsel dict.
          3. The first col => store distribution only.
          4. For each col in [1:] => fit the chosen method using the subset of rows 
             that have non-NA y.
        """
        info = loader.info()
        training_data = loader.dataframe().copy()
        if training_data.empty:
            raise ValueError("No data found => cannot train syn_seq aggregator")

        syn_order = info.get("syn_order", list(training_data.columns))
        method_map = info.get("method", {})
        varsel = info.get("variable_selection", {})

        # Force method='cart' for *cat columns, copy varsel from base col
        for col in syn_order:
            if col.endswith("_cat"):
                method_map[col] = "cart"

                base_col = col[:-4]
                if base_col in varsel:
                    varsel[col] = varsel[base_col]
                else:
                    idx = syn_order.index(col)
                    varsel[col] = syn_order[:idx]

        self._syn_order = syn_order
        self._method_map = method_map
        self._varsel = varsel

        print("[INFO] model fitting")

        # First col => store distribution for sampling
        first_col = self._syn_order[0]
        self._first_col_data = training_data[first_col].dropna().values
        print(f"Fitting '{first_col}' ... Done!")  # no model, just distribution

        # For each subsequent col
        for i, col in enumerate(self._syn_order[1:], start=1):
            chosen_m = self._method_map.get(col, "cart")
            preds_list = self._varsel.get(col, self._syn_order[:i])

            y = training_data[col].values
            X = training_data[preds_list].values

            # drop rows with NaN y
            mask = ~pd.isna(y)
            X_, y_ = X[mask], y[mask]

            print(f"Fitting '{col}' ... ", end="", flush=True)
            self._col_models[col] = self._fit_col(col, y_, X_, chosen_m)
            print("Done!")

        self._model_trained = True
        return self

    def _fit_col(self, colname: str, y: np.ndarray, X: np.ndarray, method_name: str) -> Dict[str, Any]:
        """
        Invoke the syn_* function from METHOD_MAP to train a model for col.
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
        max_iter_rules: int = 10
    ) -> pd.DataFrame:
        """
        Column-by-column generation. The first col is sampled from the real distribution, 
        then each subsequent col is generated from the fitted model. 
        If user-supplied 'rules' => re-generate violating rows up to max_iter_rules times, 
        else set them to NaN.

        Returns an "encoded" DataFrame of shape (nrows, len(_syn_order)).
        """
        if not self._model_trained:
            raise RuntimeError("Must fit the aggregator before generating")

        if nrows <= 0:
            return pd.DataFrame(columns=self._syn_order)

        np.random.seed(self.random_state)
        gen_df = pd.DataFrame(index=range(nrows))

        # First col
        first_col = self._syn_order[0]
        if self._first_col_data is not None and len(self._first_col_data) > 0:
            gen_df[first_col] = np.random.choice(self._first_col_data, size=nrows, replace=True)
        else:
            gen_df[first_col] = 0
        print(f"Generating '{first_col}' ... Done!")

        # Subsequent columns
        for col in self._syn_order[1:]:
            chosen_m = self._method_map.get(col, "cart")
            idx = self._syn_order.index(col)
            preds_list = self._varsel.get(col, self._syn_order[:idx])

            # create predictor matrix from prior columns
            Xsyn = gen_df[preds_list].values
            # generate
            ysyn = self._generate_col(col, chosen_m, Xsyn)
            gen_df[col] = ysyn

            # if user rules => re-check
            if rules and col in rules:
                tries = 0
                while True:
                    viol_idx = check_rules_violation(gen_df, col, rules)
                    if len(viol_idx) == 0:
                        break
                    if tries >= max_iter_rules:
                        gen_df.loc[viol_idx, col] = np.nan
                        print(f"[WARN] {col}: could not satisfy rules after {max_iter_rules} tries => set them to NaN.")
                        break
                    # re-generate for those violation rows
                    Xsub = gen_df.loc[viol_idx, preds_list].values
                    ysub = self._generate_col(col, chosen_m, Xsub)
                    gen_df.loc[viol_idx, col] = ysub
                    tries += 1

            print(f"Generating '{col}' ... Done!")

        return gen_df

    def _generate_col(self, colname: str, method_name: str, Xsyn: np.ndarray) -> np.ndarray:
        """
        Use the stored fitted model to generate the col's values, or fallback sampling 
        if it's the first col or no model found.
        """
        if colname not in self._col_models:
            # fallback => sample from first_col distribution
            if self._first_col_data is not None and len(self._first_col_data) > 0:
                return np.random.choice(self._first_col_data, size=len(Xsyn), replace=True)
            else:
                return np.zeros(len(Xsyn))

        fit_info = self._col_models[colname]
        gen_func = METHOD_MAP[fit_info["name"]][1]
        return gen_func(fit_info["fitted_model"], Xsyn)
