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


###############################################
# Define which methods are allowed for numeric vs. category
###############################################
ALLOWED_NUMERIC_METHODS = {
    "norm",
    "lognorm",
    "pmm",
    "cart",
    "ctree",
    "rf",
    "random",
    "swr",
    # possibly "polyreg", "logreg" if numeric is strictly binary, etc.
}

ALLOWED_CATEGORY_METHODS = {
    "cart",
    "ctree",
    "rf",
    "logreg",   # if user wants logistic for binary categories
    "polyreg",  # multi-category
    "random",
    "swr",
}


class Syn_Seq:
    """
    A sequential-synthesis aggregator that:

      1) If user_custom => we call loader.update_user_custom(user_custom)
         so that 'syn_order', 'method', 'variable_selection' are updated.
      2) We do encoded_loader, enc_dict = loader.encode(...) => returns a new encoded loader
         which has:
           - possible _cat columns for special values
           - updated method dictionary
           - updated variable_selection
           - updated col_type
      3) We partial-fit each splitted column in a hidden order (any X_cat first => forced "cart" method),
         then the main col => user-chosen or fallback. We also check col_type (numeric vs. category)
         to ensure the chosen method is appropriate. If not, we fallback.
      4) generate(...) => synthesizes col-by-col in that hidden order, optionally applying constraints.
         If strict => multiple attempts until we gather enough rows passing constraints.
      5) decode => final result is a pd.DataFrame with only the original columns (no _cat).
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
            strict: if True => repeated attempts to meet constraints
            sampling_patience: max tries if strict
            default_first_method: fallback if the first column has no method
            default_other_method: fallback for subsequent columns if user didn't specify
            seq_id_col, seq_time_col: for sequence-based constraints if needed
        """
        self.random_state = random_state
        self.strict = strict
        self.sampling_patience = sampling_patience
        self.default_first_method = default_first_method
        self.default_other_method = default_other_method
        self.seq_id_col = seq_id_col
        self.seq_time_col = seq_time_col

        # We'll store partial fits: splitted_col => {method, predictors, fit_info}
        self._col_models: Dict[str, Dict[str, Any]] = {}
        self._model_trained = False

        # We'll store enc_dict from loader.encode(...) so we can decode after generation
        self._enc_dict: Dict[str, Any] = {}

    def fit(
        self,
        loader: Any,  # typically a Syn_SeqDataLoader
        user_custom: Optional[dict] = None,
        *args,
        **kwargs
    ) -> "Syn_Seq":
        """
        1) Merge user_custom => loader.update_user_custom(...)
        2) encode => splitted loader => new df with _cat columns => encoded_loader
        3) partial-fit each splitted column, forcing 'cart' for *cat columns,
           adjusting method based on col_type if there's a mismatch
        4) store partial fits
        """
        # (1) incorporate user overrides
        if user_custom:
            loader.update_user_custom(user_custom)

        # (2) encode => splitted columns
        encoded_loader, enc_dict = loader.encode(encoders=None)
        self._enc_dict = enc_dict

        # get references
        df_encoded = encoded_loader.dataframe()
        if df_encoded.empty:
            raise ValueError("No data after encoding => cannot train on empty DataFrame.")

        syn_order = encoded_loader.syn_order            # e.g. ["age","sex","bmi","bp","target"]
        method_dict = getattr(encoded_loader, "_method", {})  # user might store it here
        varsel_df = getattr(encoded_loader._encoder, "variable_selection_", None)
        if varsel_df is None:
            varsel_df = pd.DataFrame(0, index=syn_order, columns=syn_order)

        # fallback if user didn't specify a method => default
        final_method: Dict[str, str] = {}
        for i, col_name in enumerate(syn_order):
            if col_name not in method_dict:
                final_method[col_name] = self.default_first_method if i == 0 else self.default_other_method
            else:
                final_method[col_name] = method_dict[col_name]

        # read col_type from encoded_loader
        col_type_map = getattr(encoded_loader, "col_type", {})

        # now do a splitted_map => e.g. "bp" => ["bp_cat","bp"], etc.
        splitted_map: Dict[str, List[str]] = {c: [] for c in syn_order}
        for c_enc in df_encoded.columns:
            base_c = c_enc[:-4] if c_enc.endswith("_cat") else c_enc
            splitted_map.setdefault(base_c, []).append(c_enc)

        # build final "fit_order" => cat first => forced 'cart'
        fit_order: List[str] = []
        method_map: Dict[str, str] = {}

        for i, base_col in enumerate(syn_order):
            sub_cols = splitted_map.get(base_col, [])
            cat_cols = [x for x in sub_cols if x.endswith("_cat")]
            main_cols = [x for x in sub_cols if not x.endswith("_cat")]

            # (A) cat columns => forced "cart"
            for sc in cat_cols:
                fit_order.append(sc)
                method_map[sc] = "cart"

            # (B) main col => check col_type => fix method if mismatch
            chosen_m = final_method.get(base_col, self.default_other_method)
            declared_t = col_type_map.get(base_col, "category")  # default "category"

            # if numeric => ensure chosen_m in ALLOWED_NUMERIC_METHODS
            # if category => ensure chosen_m in ALLOWED_CATEGORY_METHODS
            cfix = chosen_m.strip().lower()
            if declared_t.lower() == "numeric":
                if cfix not in ALLOWED_NUMERIC_METHODS:
                    # fallback => e.g. "norm"
                    fallback_m = self.default_first_method if i == 0 else "norm"
                    print(f"[TYPE-CHECK] '{base_col}' is numeric but method '{cfix}' invalid => fallback '{fallback_m}'")
                    cfix = fallback_m
            else:
                # category => ensure cfix in ALLOWED_CATEGORY_METHODS
                if cfix not in ALLOWED_CATEGORY_METHODS:
                    # fallback => "cart"
                    print(f"[TYPE-CHECK] '{base_col}' is category but method '{cfix}' invalid => fallback 'cart'")
                    cfix = "cart"

            # store final
            for sc in main_cols:
                fit_order.append(sc)
                method_map[sc] = cfix

        # leftover columns not in syn_order => fallback
        leftover = [c for c in df_encoded.columns if c not in fit_order]
        for c in leftover:
            fit_order.append(c)
            method_map[c] = self.default_other_method

        # Logging
        print("[INFO] final synthesis:")
        print(f"  - syn_order: {syn_order}")
        print(f"  - method: {final_method}")
        print("  - variable_selection_:")
        print(varsel_df)
        print("\n[INFO] model fitting")

        self._col_models.clear()
        for col_enc in fit_order:
            base_c = col_enc[:-4] if col_enc.endswith("_cat") else col_enc
            chosen_m = method_map[col_enc]

            # gather predictors from varsel_df
            preds_list = []
            if base_c in varsel_df.index:
                row_mask = varsel_df.loc[base_c] == 1
                preds_list = varsel_df.columns[row_mask].tolist()
                # if base is not used => also remove base_cat
                preds_list = [
                    p for p in preds_list
                    if not (p.endswith("_cat") and p[:-4] not in preds_list)
                ]

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

    def _fit_single_column(self, y: np.ndarray, X: np.ndarray, method_name: str) -> Dict[str, Any]:
        """
        We store partial info for each col => { "type":..., "obs_y":..., "obs_X":... }
        used at generation time to call e.g. syn_cart(...) or syn_norm(...)
        """
        m = method_name.strip().lower()
        if m in {
            "cart", "ctree", "rf", "norm", "lognorm", "pmm", "logreg", "polyreg"
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
        encoded_loader: Any,  # same type of Syn_SeqDataLoader
        constraints: Optional[Dict[str, List[Any]]] = None,
        *args,
        **kwargs
    ) -> pd.DataFrame:
        """
        1) generate splitted columns in the fitted order
        2) apply constraints if any
        3) decode => revert to original columns
        4) return final DataFrame
        """
        if not self._model_trained:
            raise RuntimeError("Must fit(...) aggregator before .generate().")

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

        # decode => final
        tmp_loader = encoded_loader.decorate(df_synth)
        final_loader = tmp_loader.decode(self._enc_dict)
        return final_loader.dataframe()

    def _attempt_strict_generation(self, count: int, syn_constraints: Syn_SeqConstraints) -> pd.DataFrame:
        """
        repeated tries => gather enough rows that pass constraints
        do direct substitution for '=' => then match => drop_duplicates => accumulate
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
        single pass => produce splitted columns in fit_order
        """
        fit_order = list(self._col_models.keys())  # splitted columns
        syn_df = pd.DataFrame(index=range(count))

        print("")
        for col_enc in fit_order:
            info = self._col_models[col_enc]
            method_name = info["method"]
            preds = info["predictors"]
            fit_data = info["fit_info"]

            Xp = syn_df[preds].values if preds else np.zeros((count, 0))
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
        Dispatch to the specialized column-synthesis function
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
                random_state=self.random_state,
            )
            return pd.Series(result["res"])
        elif m == "random":
            result = syn_random(
                y=y_obs,
                X=None,
                Xp=np.zeros((count,1)),
                random_state=self.random_state,
            )
            return pd.Series(result["res"])
        else:
            # fallback => random
            fallback = syn_random(
                y=y_obs,
                X=None,
                Xp=np.zeros((count,1)),
                random_state=self.random_state,
            )
            return pd.Series(fallback["res"])
