# File: syn_seq.py

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

# If your constraints class is in syn_seq_constraints.py
from synthcity.plugins.core.models.syn_seq.syn_seq_constraints import Syn_SeqConstraints

# The column-by-column methods
from synthcity.plugins.core.models.syn_seq.methods.cart import syn_cart
from synthcity.plugins.core.models.syn_seq.methods.ctree import syn_ctree
from synthcity.plugins.core.models.syn_seq.methods.logreg import syn_logreg
from synthcity.plugins.core.models.syn_seq.methods.norm import syn_norm, syn_lognorm
from synthcity.plugins.core.models.syn_seq.methods.pmm import syn_pmm
from synthcity.plugins.core.models.syn_seq.methods.polyreg import syn_polyreg
from synthcity.plugins.core.models.syn_seq.methods.rf import syn_rf
from synthcity.plugins.core.models.syn_seq.methods.misc import syn_random, syn_swr

class Syn_Seq:
    """
    A sequential-synthesis aggregator.

    Steps:
      1) If user_custom => call loader.update_user_custom(user_custom).
         That might change loader.syn_order, loader.method, loader.variable_selection, etc.
      2) aggregator calls encoded_loader, enc_dict = loader.encode(...) => new loader with _cat columns
      3) aggregator partial-fits each splitted column in a hidden order:
         - any "bp_cat" or "target_cat" columns first (forced "cart" method),
         - then the main column "bp" or "target" with the user-chosen or fallback method.
      4) generate(...) => synthesizes columns in that order, optionally applying constraints.
      5) decode => final output is a plain DataFrame with only original columns (no _cat).
    """

    def __init__(
        self,
        random_state: int = 0,
        strict: bool = True,
        sampling_patience: int = 100,
        default_first_method: str = "swr",
        default_other_method: str = "cart",
        seq_id_col: str = "seq_id",
        seq_time_col: str = "seq_time_id",
    ):
        """
        Args:
            random_state: reproducibility
            strict: if True => repeated tries to meet constraints
            sampling_patience: max tries if strict
            default_first_method: fallback if the first column has no method
            default_other_method: fallback for subsequent columns if no user-specified method
            seq_id_col, seq_time_col: for group-based constraints if needed
        """
        self.random_state = random_state
        self.strict = strict
        self.sampling_patience = sampling_patience
        self.default_first_method = default_first_method
        self.default_other_method = default_other_method

        self.seq_id_col = seq_id_col
        self.seq_time_col = seq_time_col

        # We'll store partial fits in a dict: col_enc => { "method":..., "predictors":..., "fit_info":... }
        self._col_models: Dict[str, Dict[str, Any]] = {}
        self._model_trained = False

        # We'll store the encoding dictionary from loader.encode(...) so we can decode automatically
        self._enc_dict: Dict[str, Any] = {}

    def fit(
        self,
        loader: Any,  # typically a Syn_SeqDataLoader
        user_custom: Optional[dict] = None,
        *args,
        **kwargs
    ) -> "Syn_Seq":
        """
        1) Merge user_custom into loader (if any).
        2) loader.encode(...) => returns encoded_loader, enc_dict
        3) partial-fit each splitted column (including _cat).
        4) store partial fits in self._col_models
        """
        # (1) incorporate any user overrides
        if user_custom:
            loader.update_user_custom(user_custom)

        # (2) encode => splitted loader with `_cat` columns
        #    We do NOT do partial sampling => use entire dataset
        encoded_loader, enc_dict = loader.encode(encoders=None)
        self._enc_dict = enc_dict  # so we can decode after generation

        df_encoded = encoded_loader.dataframe()
        if df_encoded.empty:
            raise ValueError("No data after encoding => cannot train on empty DataFrame.")

        syn_order = encoded_loader.syn_order  # e.g. ["age","sex","bmi","bp","target"]
        method_dict = encoded_loader.method   # e.g. {"bp":"polyreg","bmi":"cart",...}
        varsel_df = encoded_loader.variable_selection  # k-by-k DataFrame

        # 2a) figure out final method for each original col in syn_order
        final_method: Dict[str, str] = {}
        for i, col_name in enumerate(syn_order):
            if col_name not in method_dict:
                # fallback
                final_method[col_name] = (
                    self.default_first_method if i == 0 else self.default_other_method
                )
            else:
                final_method[col_name] = method_dict[col_name]

        # 2b) build a splitted map => for each base col => a list of splitted columns [col_cat, col]
        splitted_map: Dict[str, List[str]] = {c: [] for c in syn_order}
        for c_enc in df_encoded.columns:
            base_c = c_enc[:-4] if c_enc.endswith("_cat") else c_enc
            if base_c not in splitted_map:
                splitted_map[base_c] = [c_enc]
            else:
                splitted_map[base_c].append(c_enc)

        # 3) flatten them => cat first => forced "cart", then main col => user or fallback
        fit_order: List[str] = []
        method_map: Dict[str, str] = {}

        for i, base_col in enumerate(syn_order):
            sub_cols = splitted_map.get(base_col, [])
            cat_cols = [x for x in sub_cols if x.endswith("_cat")]
            main_cols = [x for x in sub_cols if not x.endswith("_cat")]
            chosen = final_method[base_col]

            # cat columns => forced "cart"
            for sc in cat_cols:
                fit_order.append(sc)
                method_map[sc] = "cart"

            # main col => chosen/fallback
            for sc in main_cols:
                fit_order.append(sc)
                method_map[sc] = chosen

        # leftover columns not in syn_order
        leftover = [c for c in df_encoded.columns if c not in fit_order]
        for c in leftover:
            fit_order.append(c)
            method_map[c] = self.default_other_method  # fallback

        # Logging
        print("[INFO] final synthesis:")
        print(f"  - syn_order: {syn_order}")
        print(f"  - method: {final_method}")
        print("  - variable_selection_:")
        print(varsel_df)
        print("[INFO] model fitting")

        # 4) partial-fit each splitted column
        self._col_models.clear()
        for col_enc in fit_order:
            base_c = col_enc[:-4] if col_enc.endswith("_cat") else col_enc
            chosen_m = method_map[col_enc]

            # get predictor set from varsel_df
            if base_c in varsel_df.index:
                pred_mask = varsel_df.loc[base_c] == 1
                preds_list = varsel_df.columns[pred_mask].tolist()
                # If user doesn't want "bp", also exclude "bp_cat"
                # e.g. if "bp" not in preds, then remove "bp_cat"
                preds_list = [
                    p for p in preds_list
                    if not (p.endswith("_cat") and p[:-4] not in preds_list)
                ]
            else:
                preds_list = []

            y = df_encoded[col_enc].values
            X = df_encoded[preds_list].values if preds_list else np.zeros((len(y), 0))

            print(f"Fitting '{col_enc}' ... ", end="")
            fit_info = self._fit_single_column(y, X, chosen_m)
            self._col_models[col_enc] = {
                "method": chosen_m,
                "predictors": preds_list,
                "fit_info": fit_info,
            }
            print("Done!")

        self._model_trained = True
        return self

    def _fit_single_column(
        self, y: np.ndarray, X: np.ndarray, method_name: str
    ) -> Dict[str, Any]:
        """
        We store partial info (obs_y, obs_X, method_type).
        e.g. { "type": "cart", "obs_y": y, "obs_X": X }
        We'll use it at generation time.
        """
        m = method_name.strip().lower()
        if m in {
            "cart", "ctree", "rf", "norm", "lognorm",
            "pmm", "logreg", "polyreg"
        }:
            return {"type": m, "obs_y": y, "obs_X": X}
        elif m == "swr":
            return {"type": "swr", "obs_y": y}
        elif m == "random":
            return {"type": "random", "obs_y": y}
        else:
            # fallback => random
            return {"type": "random", "obs_y": y}

    def generate(
        self,
        count: int,
        encoded_loader: Any,
        constraints: Optional[Dict[str, List[Any]]] = None,
        *args,
        **kwargs
    ) -> pd.DataFrame:
        """
        1) Generate splitted columns
        2) If constraints => strict or single-pass filtering
        3) decode => revert to original columns
        4) Return final DataFrame
        """
        if not self._model_trained:
            raise RuntimeError("Must fit(...) aggregator before generate().")

        # Build constraints object if dictionary is provided
        syn_constraints = None
        if constraints is not None:
            syn_constraints = Syn_SeqConstraints(chained_rules=constraints)

        if self.strict and syn_constraints:
            df_synth = self._attempt_strict_generation(count, syn_constraints)
        else:
            df_synth = self._generate_once(count)
            if syn_constraints:
                df_synth = syn_constraints.correct_equals(df_synth)
                df_synth = syn_constraints.match(df_synth)

        # decode => get only original user columns
        tmp_loader = encoded_loader.decorate(df_synth)
        final_loader = tmp_loader.decode(self._enc_dict)
        return final_loader.dataframe()

    def _attempt_strict_generation(
        self,
        count: int,
        syn_constraints: Syn_SeqConstraints
    ) -> pd.DataFrame:
        """
        repeated attempts => gather rows meeting constraints
        if equality => direct substitution first => then filter => drop_duplicates
        """
        result_df = pd.DataFrame()
        tries = 0
        while len(result_df) < count and tries < self.sampling_patience:
            tries += 1
            chunk = self._generate_once(count)
            chunk = syn_constraints.correct_equals(chunk)
            chunk = syn_constraints.match(chunk)
            chunk = chunk.drop_duplicates()
            result_df = pd.concat([result_df, chunk], ignore_index=True)

        return result_df.head(count)

    def _generate_once(self, count: int) -> pd.DataFrame:
        """
        single pass => produce splitted columns in fit_order (the order of self._col_models)
        """
        col_list = list(self._col_models.keys())
        syn_df = pd.DataFrame(index=range(count))

        print("")
        for col_enc in col_list:
            info = self._col_models[col_enc]
            method_name = info["method"]
            preds = info["predictors"]
            fit_data = info["fit_info"]

            if preds:
                Xp = syn_df[preds].values
            else:
                Xp = np.zeros((count, 0))

            print(f"Generating '{col_enc}' ... ", end="")
            new_vals = self._generate_single_column(method_name, fit_data, Xp, count)
            syn_df[col_enc] = new_vals
            print("Done!")

        return syn_df

    def _generate_single_column(
        self,
        method: str,
        fit_info: Dict[str, Any],
        Xp: np.ndarray,
        count: int
    ) -> pd.Series:
        """
        For each method type, dispatch to the corresponding generator function
        """
        m = method.strip().lower()
        y_obs = fit_info.get("obs_y")
        X_obs = fit_info.get("obs_X")

        if m == "cart":
            result = syn_cart(y=y_obs, X=X_obs, Xp=Xp, random_state=self.random_state)
            return pd.Series(result["res"])
        elif m == "ctree":
            result = syn_ctree(y=y_obs, X=X_obs, Xp=Xp, random_state=self.random_state)
            return pd.Series(result["res"])
        elif m == "rf":
            result = syn_rf(y=y_obs, X=X_obs, Xp=Xp, random_state=self.random_state)
            return pd.Series(result["res"])
        elif m == "norm":
            result = syn_norm(y=y_obs, X=X_obs, Xp=Xp, random_state=self.random_state)
            return pd.Series(result["res"])
        elif m == "lognorm":
            result = syn_lognorm(y=y_obs, X=X_obs, Xp=Xp, random_state=self.random_state)
            return pd.Series(result["res"])
        elif m == "pmm":
            result = syn_pmm(y=y_obs, X=X_obs, Xp=Xp, random_state=self.random_state)
            return pd.Series(result["res"])
        elif m == "logreg":
            result = syn_logreg(y=y_obs, X=X_obs, Xp=Xp, random_state=self.random_state)
            return pd.Series(result["res"])
        elif m == "polyreg":
            result = syn_polyreg(y=y_obs, X=X_obs, Xp=Xp, random_state=self.random_state)
            return pd.Series(result["res"])
        elif m == "swr":
            result = syn_swr(
                y=y_obs,
                X=None,
                Xp=np.zeros((count,1)),
                random_state=self.random_state
            )
            return pd.Series(result["res"])
        elif m == "random":
            result = syn_random(
                y=y_obs,
                X=None,
                Xp=np.zeros((count,1)),
                random_state=self.random_state
            )
            return pd.Series(result["res"])
        else:
            # fallback => random
            fallback = syn_random(
                y=y_obs,
                X=None,
                Xp=np.zeros((count,1)),
                random_state=self.random_state
            )
            return pd.Series(fallback["res"])
