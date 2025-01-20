# File: syn_seq.py

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from synthcity.logger import info, warning
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.schema import Schema

# The column-fitting and column-generating methods from your submodule
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
# We define a helper function that checks whether a row violates a set of rules.
# e.g. rules = { "target": [("bmi", ">", 0.15), ("target", ">", 0)] }
# If any rule is violated, that row must be regenerated.
# ------------------------------------------------------------------

def check_rules_violation(
    df: pd.DataFrame,
    target_col: str,
    rules_dict: Dict[str, List[Tuple[str, str, Any]]]
) -> pd.Index:
    """
    For the current 'target_col', check if it has any rules from rules_dict.
    If rules_dict[target_col] = [("some_col","=",val), ("other_col", ">", val2), ...],
    we interpret them as conditions that must ALL be satisfied.

    We return the row-indices that FAIL these conditions.
    """
    if target_col not in rules_dict:
        return pd.Index([])  # no rules => no violation

    sub_rules = rules_dict[target_col]
    mask_valid = pd.Series(True, index=df.index)

    for (col_feat, operator, val) in sub_rules:
        if col_feat not in df.columns:
            # skip if that feature doesn't exist in df
            continue

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
            # unrecognized operator => skip
            cond = pd.Series(True, index=df.index)

        mask_valid &= cond

    # Return indices that fail (i.e. not valid)
    return df.index[~mask_valid]

# ------------------------------------------------------------------
# Map method_name => (syn_func, generate_func) from your submodule,
# so we can dynamically choose the correct approach per column.
# ------------------------------------------------------------------

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
    Column-by-column sequential regression aggregator.

    On .fit_col(...), we parse relevant order/method/varsel from the DataLoader info
    and train each column in sequence.

    On .generate_col(...), we sample from the aggregator column by column, optionally
    applying user rules or constraints.
    """

    def __init__(
        self,
        random_state: int = 0,
        strict: bool = True,
        sampling_patience: int = 100
    ):
        """
        Args:
            random_state: random seed
            strict: if True, we do re-check constraints (or user rules)
            sampling_patience: max tries for re-generation (not used here but could be).
        """
        self.random_state = random_state
        self.strict = strict
        self.sampling_patience = sampling_patience

        self._model_trained = False
        self._syn_order: List[str] = []
        self._method_map: Dict[str, str] = {}
        self._varsel: Dict[str, List[str]] = {}
        self._col_models: Dict[str, Dict[str, Any]] = {}

        # For the first column, we store real distribution
        self._first_col_data: Optional[np.ndarray] = None

    def fit_col(self, loader: DataLoader, *args, **kwargs) -> "Syn_Seq":
        """
        Fit column-by-column from the "encoded" DataLoader.

        Steps:
          - read info: syn_order, method, variable_selection
          - store the distribution of the first column
          - for each subsequent column, train with the selected method
        """
        info_dict = loader.info()
        training_data = loader.dataframe().copy()
        if training_data.empty:
            raise ValueError("No data => cannot fit Syn_Seq aggregator")

        self._syn_order = info_dict.get("syn_order", list(training_data.columns))
        self._method_map = info_dict.get("method", {})
        self._varsel = info_dict.get("variable_selection", {})

        # Force columns ending with "_cat" => method='cart'
        # also replicate varsel if needed
        for col in self._syn_order:
            if col.endswith("_cat"):
                self._method_map[col] = "cart"
                base_col = col[:-4]
                if base_col in self._varsel:
                    self._varsel[col] = self._varsel[base_col]
                else:
                    idx = self._syn_order.index(col)
                    self._varsel[col] = self._syn_order[:idx]

        info("[INFO] Syn_Seq aggregator: fitting columns...")

        # 1) First col => store real distribution
        first_col = self._syn_order[0]
        self._first_col_data = training_data[first_col].dropna().values
        info(f"Fitting '{first_col}' => stored distribution from real data. Done.")

        # 2) Fit each subsequent column
        np.random.seed(self.random_state)
        for i, col in enumerate(self._syn_order[1:], start=1):
            method_name = self._method_map.get(col, "cart")
            preds_list = self._varsel.get(col, self._syn_order[:i])

            # Y is the column data, X are the preceding columns
            y = training_data[col].values
            X = training_data[preds_list].values

            # drop rows with NaN in y
            mask = ~pd.isna(y)
            X_ = X[mask]
            y_ = y[mask]

            info(f"Fitting '{col}' with '{method_name}' ... ", end="", flush=True)
            self._col_models[col] = self._fit_single_col(method_name, X_, y_)
            info("Done!")

        self._model_trained = True
        return self

    def _fit_single_col(self, method_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Trains the chosen method for one column
        """
        fit_func, _ = METHOD_MAP[method_name]
        model = fit_func(y, X, random_state=self.random_state)
        return {
            "name": method_name,
            "fitted_model": model,
        }

    def generate_col(
        self,
        nrows: int,
        rules: Optional[Dict[str, List[Tuple[str, str, Any]]]] = None,
        max_iter_rules: int = 10
    ) -> pd.DataFrame:
        """
        Generate `nrows` rows, column by column.

        If `rules` is given, after generating each column,
        we check for rows that fail the rule => re-generate up to max_iter_rules attempts.
        """
        if not self._model_trained:
            raise RuntimeError("Syn_Seq aggregator not yet fitted")

        if nrows <= 0:
            return pd.DataFrame(columns=self._syn_order)

        np.random.seed(self.random_state)
        gen_df = pd.DataFrame(index=range(nrows))

        # 1) Generate the first col from real distribution
        first_col = self._syn_order[0]
        if self._first_col_data is not None and len(self._first_col_data) > 0:
            gen_df[first_col] = np.random.choice(self._first_col_data, size=nrows, replace=True)
        else:
            gen_df[first_col] = 0
        info(f"Generating '{first_col}' => done.")

        # 2) For each subsequent col, generate from fitted model
        for col in self._syn_order[1:]:
            method_name = self._method_map.get(col, "cart")
            idx = self._syn_order.index(col)
            preds_list = self._varsel.get(col, self._syn_order[:idx])

            # Xsyn are the preceding columns
            Xsyn = gen_df[preds_list].values
            ysyn = self._generate_single_col(method_name, Xsyn, col)
            gen_df[col] = ysyn

            # If user rules => re-generate any violating rows
            if rules and col in rules:
                tries = 0
                while True:
                    viol_idx = check_rules_violation(gen_df, col, rules)
                    if viol_idx.empty:
                        break
                    if tries >= max_iter_rules:
                        warning(
                            f"[WARN] {col}: cannot satisfy rules after {max_iter_rules} tries => set them to NaN"
                        )
                        gen_df.loc[viol_idx, col] = np.nan
                        break
                    # re-generate only the violating subset
                    Xsub = gen_df.loc[viol_idx, preds_list].values
                    ysub = self._generate_single_col(method_name, Xsub, col)
                    gen_df.loc[viol_idx, col] = ysub
                    tries += 1

            info(f"Generating '{col}' => done.")

        return gen_df

    def _generate_single_col(self, method_name: str, Xsyn: np.ndarray, col: str) -> np.ndarray:
        """
        Use the fitted model for col, or fallback to first-col distribution if not found
        """
        if col not in self._col_models:
            # fallback => first col distribution
            if self._first_col_data is not None and len(self._first_col_data) > 0:
                return np.random.choice(self._first_col_data, size=len(Xsyn), replace=True)
            else:
                return np.zeros(len(Xsyn))

        fit_info = self._col_models[col]
        gen_func = METHOD_MAP[fit_info["name"]][1]
        return gen_func(fit_info["fitted_model"], Xsyn)
