# File: syn_seq.py

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from synthcity.logger import info, warning
from synthcity.plugins.core.dataloader import DataLoader

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
# Utility function to check if rows violate user-defined rules
# ------------------------------------------------------------------
def check_rules_violation(
    df: pd.DataFrame,
    target_col: str,
    rules_dict: Dict[str, List[Tuple[str, str, Any]]]
) -> pd.Index:
    """
    For the current 'target_col', check if it has any rules from rules_dict.

    Example of rules:
        rules = {
          "target": [
              ("bmi", ">", 0.15),
              ("target", "=", 0)
          ]
        }

    Each tuple => (col_feat, operator, val).
    All conditions must be satisfied for a row to be valid. If any condition fails,
    that row is considered a violation.

    Returns:
        pd.Index of row indices that FAIL these conditions.
    """
    if target_col not in rules_dict:
        return pd.Index([])  # no rules => no violations for this column

    sub_rules = rules_dict[target_col]
    mask_valid = pd.Series(True, index=df.index)

    for (col_feat, operator, val) in sub_rules:
        if col_feat not in df.columns:
            # skip if that feature doesn't exist
            continue

        col_data = df[col_feat]

        if operator in ["=", "=="]:
            cond = (col_data == val) | col_data.isna()
        elif operator in [">"]:
            cond = (col_data > val) | col_data.isna()
        elif operator in [">="]:
            cond = (col_data >= val) | col_data.isna()
        elif operator in ["<"]:
            cond = (col_data < val) | col_data.isna()
        elif operator in ["<="]:
            cond = (col_data <= val) | col_data.isna()
        else:
            # unrecognized operator => skip or treat as always true
            cond = pd.Series(True, index=df.index)

        mask_valid &= cond

    # Return the rows that are NOT valid
    return df.index[~mask_valid]


# ------------------------------------------------------------------
# Map method_name => (syn_func, generate_func) 
# so we can dynamically select how each column is trained & generated.
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
    A column-by-column aggregator.

    On .fit_col(...):
       - parse info from DataLoader (syn_order, method, varsel, etc.)
       - store distribution of the first column for direct sampling
       - train an aggregator for each subsequent column

    On .generate_col(...):
       - sample columns in the same order, optionally re-check user rules
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
            strict: if True, re-check constraints (or user rules) during generation
            sampling_patience: max attempts for re-generation if constraints fail
        """
        self.random_state = random_state
        self.strict = strict
        self.sampling_patience = sampling_patience

        # Internal state
        self._model_trained = False
        self._syn_order: List[str] = []
        self._method_map: Dict[str, str] = {}
        self._varsel: Dict[str, List[str]] = {}
        self._col_models: Dict[str, Dict[str, Any]] = {}

        # Real distribution for the first column
        self._first_col_data: Optional[np.ndarray] = None

    def fit_col(self, loader: DataLoader, *args, **kwargs) -> "Syn_Seq":
        """
        Fit column-by-column from the "encoded" DataLoader.

        Steps:
        1) read info: syn_order, method, variable_selection
        2) store real distribution for first column => for sampling
        3) for each subsequent column, pick aggregator => train with that method

        Modification:
        - If the column's 'converted_type' is 'category', skip the np.isnan check on X
            to avoid TypeError for object/string dtypes.
        """
        # 1) Gather the relevant metadata
        info_dict = loader.info()
        training_data = loader.dataframe().copy()
        if training_data.empty:
            raise ValueError("No data => cannot fit Syn_Seq aggregator")

        self._syn_order = info_dict.get("syn_order", list(training_data.columns))
        self._method_map = info_dict.get("method", {})
        self._varsel = info_dict.get("variable_selection", {})
        conv_type_map = info_dict.get("converted_type", {})  # <-- for checking 'category'

        # 2) Force columns ending with "_cat" => aggregator "cart" if not set
        for col in self._syn_order:
            if col.endswith("_cat"):
                self._method_map[col] = "cart"
                base_col = col[:-4]
                if base_col in self._varsel:
                    # replicate varsel
                    self._varsel[col] = self._varsel[base_col]
                else:
                    idx = self._syn_order.index(col)
                    self._varsel[col] = self._syn_order[:idx]

        info("[INFO] Syn_Seq aggregator: fitting columns...")

        # 3) The first column: store its distribution for direct sampling
        first_col = self._syn_order[0]
        # ignoring rows where first_col is missing
        self._first_col_data = training_data[first_col].dropna().values
        info(f"Fitting '{first_col}' => stored distribution from real data. Done.")

        # 4) Train aggregator column-by-column for subsequent columns
        np.random.seed(self.random_state)
        for i, col in enumerate(self._syn_order[1:], start=1):
            method_name = self._method_map.get(col, "cart")
            preds_list = self._varsel.get(col, self._syn_order[:i])

            # Y = the column's data; X = preceding columns
            y = training_data[col].values
            X = training_data[preds_list].values

            # Check if this column's converted_type is "category"
            col_ctype = conv_type_map.get(col, "")

            # For Y, we can still drop NaNs:

            if col_ctype == "category":
                # If it's a category column, skip np.isnan(...) on X entirely
                mask_x = np.ones(len(X), dtype=bool)
                mask_y = np.ones(y, dtype=bool)
            else:
                mask_x = ~np.isnan(X).any(axis=1)
                mask_y = ~pd.isna(y)

            # Combine the masks
            mask = mask_y & mask_x
            X_ = X[mask]
            y_ = y[mask]

            info(f"Fitting '{col}' with '{method_name}' ... ", end="", flush=True)
            self._col_models[col] = self._fit_single_col(method_name, X_, y_)
            info("Done!")

        self._model_trained = True
        return self


    def _fit_single_col(self, method_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Trains the chosen method for one column; returns a dict:
            {
              "name": <method_name>,
              "fitted_model": <actual model object>
            }
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

        If `rules` is given, after generating each column, we check for any row that fails
        => we re-generate it up to `max_iter_rules` times or set it to NaN if impossible.
        """
        if not self._model_trained:
            raise RuntimeError("Syn_Seq aggregator not yet fitted")

        if nrows <= 0:
            return pd.DataFrame(columns=self._syn_order)

        np.random.seed(self.random_state)
        gen_df = pd.DataFrame(index=range(nrows))

        # (1) Generate the first column from real distribution
        first_col = self._syn_order[0]
        if self._first_col_data is not None and len(self._first_col_data) > 0:
            gen_df[first_col] = np.random.choice(
                self._first_col_data, size=nrows, replace=True
            )
        else:
            gen_df[first_col] = 0
        info(f"Generating '{first_col}' => done.")

        # (2) For each subsequent column, generate using fitted model
        for col in self._syn_order[1:]:
            method_name = self._method_map.get(col, "cart")
            idx = self._syn_order.index(col)
            preds_list = self._varsel.get(col, self._syn_order[:idx])

            # gather the already generated columns
            Xsyn = gen_df[preds_list].values
            ysyn = self._generate_single_col(method_name, Xsyn, col)
            gen_df[col] = ysyn

            # If user rules => check for violations => re-generate if needed
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
                    # re-generate only those violating rows
                    Xsub = gen_df.loc[viol_idx, preds_list].values
                    ysub = self._generate_single_col(method_name, Xsub, col)
                    gen_df.loc[viol_idx, col] = ysub
                    tries += 1

            info(f"Generating '{col}' => done.")

        return gen_df

    def _generate_single_col(self, method_name: str, Xsyn: np.ndarray, col: str) -> np.ndarray:
        """
        Use the fitted model for 'col' if available, otherwise fallback
        to sampling from the first column's distribution (less ideal).
        """
        if col not in self._col_models:
            # fallback => sample from self._first_col_data
            if self._first_col_data is not None and len(self._first_col_data) > 0:
                return np.random.choice(self._first_col_data, size=len(Xsyn), replace=True)
            else:
                return np.zeros(len(Xsyn))

        # call the aggregator's 'generate' function
        fit_info = self._col_models[col]
        _, generate_func = METHOD_MAP[fit_info["name"]]
        return generate_func(fit_info["fitted_model"], Xsyn)
