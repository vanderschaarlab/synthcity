from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from synthcity.logger import info, warning
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.models.syn_seq.syn_seq_encoder import Syn_SeqEncoder
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

def check_rules_violation(
    df: pd.DataFrame,
    target_col: str,
    rules_dict: Dict[str, List[Tuple[str, str, Any]]]
) -> pd.Index:
    if target_col not in rules_dict:
        return pd.Index([])
    sub_rules = rules_dict[target_col]
    mask_valid = pd.Series(True, index=df.index)
    for (feat, operator, thresh) in sub_rules:
        if feat not in df.columns:
            continue
        col_data = df[feat]
        if operator in ["=", "=="]:
            cond = (col_data == thresh) | col_data.isna()
        elif operator in [">"]:
            cond = (col_data > thresh) | col_data.isna()
        elif operator in [">="]:
            cond = (col_data >= thresh) | col_data.isna()
        elif operator in ["<"]:
            cond = (col_data < thresh) | col_data.isna()
        elif operator in ["<="]:
            cond = (col_data <= thresh) | col_data.isna()
        else:
            cond = pd.Series(True, index=df.index)
        mask_valid &= cond
    return df.index[~mask_valid]


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
        self.random_state = random_state
        self.strict = strict
        self.sampling_patience = sampling_patience
        self._model_trained = False
        # The syn_order will be provided by the loader's info—which now includes the auto‑injected “_cat” columns.
        self._syn_order: List[str] = []
        self._method_map: Dict[str, str] = {}
        self._varsel: Dict[str, List[str]] = {}
        self._col_models: Dict[str, Dict[str, Any]] = {}
        self._first_col_data: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(self.random_state)

    def fit_col(self, loader: DataLoader, *args: Any, **kwargs: Any) -> "Syn_Seq":
        info("[INFO] Syn_Seq aggregator: fitting columns...")
        info_dict = loader.info()
        training_data = loader.dataframe().copy()
        if training_data.empty:
            raise ValueError("No data => cannot fit Syn_Seq aggregator")
        # Get the updated syn_order, method, and variable_selection from the loader's info
        self._syn_order = info_dict.get("syn_order", list(training_data.columns))
        self._method_map = info_dict.get("method", {})
        self._varsel = info_dict.get("variable_selection", {})

        for col in self._syn_order:
            if col.endswith("_cat"):
                self._method_map[col] = "cart"
                base_col = col[:-4]
                if base_col in self._varsel:
                    self._varsel[col] = self._varsel[base_col]
                else:
                    idx = self._syn_order.index(col)
                    self._varsel[col] = self._syn_order[:idx]

        first_col = self._syn_order[0]
        self._first_col_data = training_data[first_col].dropna().values
        info(f"Fitting '{first_col}' => stored distribution from real data. Done.")

        for idx, col in enumerate(self._syn_order[1:], start=1):
            method_name = self._method_map.get(col, "cart")
            predictors = self._varsel.get(col, self._syn_order[:idx])
            y = training_data[col].values
            X = training_data[predictors].values
            valid_mask = (~np.isnan(X).any(axis=1)) & (~pd.isna(y))
            X_train = X[valid_mask]
            y_train = y[valid_mask]
            info(f"Fitting '{col}' with '{method_name}' ...")
            self._col_models[col] = self._fit_single_col(method_name, X_train, y_train)
            info("Done!")
        self._model_trained = True
        return self

    def _fit_single_col(self, method_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        fit_func, _ = METHOD_MAP[method_name]
        model = fit_func(y, X, random_state=self.random_state)
        return {"name": method_name, "fitted_model": model}

    def generate_col(
        self,
        nrows: int,
        rules: Optional[Dict[str, List[Tuple[str, str, Any]]]] = None,
        max_iter_rules: int = 10
    ) -> pd.DataFrame:
        if not self._model_trained:
            raise RuntimeError("Syn_Seq aggregator not yet fitted")
        if nrows <= 0:
            return pd.DataFrame(columns=self._syn_order)

        gen_df = pd.DataFrame(index=range(nrows))
        first_col = self._syn_order[0]
        if self._first_col_data is not None and len(self._first_col_data) > 0:
            gen_df[first_col] = self._rng.choice(self._first_col_data, size=nrows, replace=True)
        else:
            gen_df[first_col] = 0
        info(f"Generating '{first_col}' => done.")

        for idx, col in enumerate(self._syn_order[1:], start=1):
            method_name = self._method_map.get(col, "cart")
            predictors = self._varsel.get(col, self._syn_order[:idx])
            Xsyn = gen_df[predictors].values
            y_syn = self._generate_single_col(method_name, Xsyn, col)
            gen_df[col] = y_syn

            if rules and col in rules:
                tries = 0
                while True:
                    viol_idx = check_rules_violation(gen_df, col, rules)
                    if viol_idx.empty:
                        break
                    if tries >= max_iter_rules:
                        warning(f"[WARN] {col}: cannot satisfy rules after {max_iter_rules} tries => set them to NaN")
                        gen_df.loc[viol_idx, col] = np.nan
                        break
                    X_sub = gen_df.loc[viol_idx, predictors].values
                    y_sub = self._generate_single_col(method_name, X_sub, col)
                    gen_df.loc[viol_idx, col] = y_sub
                    tries += 1

            info(f"Generating '{col}' => done.")

        return gen_df

    def _generate_single_col(self, method_name: str, Xsyn: np.ndarray, col: str) -> np.ndarray:
        if col not in self._col_models:
            if self._first_col_data is not None and len(self._first_col_data) > 0:
                return self._rng.choice(self._first_col_data, size=len(Xsyn), replace=True)
            else:
                return np.zeros(len(Xsyn))
        fit_info = self._col_models[col]
        _, generate_func = METHOD_MAP[fit_info["name"]]
        return generate_func(fit_info["fitted_model"], Xsyn)


plugin = Syn_Seq
