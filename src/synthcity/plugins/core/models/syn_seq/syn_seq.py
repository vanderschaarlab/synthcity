# File: syn_seq.py

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

# synergy imports
from synthcity.plugins.core.dataloader import Syn_SeqDataLoader
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.models.syn_seq.syn_seq_constraints import SynSeqConstraints

# methods for actual column-by-column synthesis
from synthcity.plugins.core.models.syn_seq.methods.cart import syn_cart
from synthcity.plugins.core.models.syn_seq.methods.ctree import syn_ctree
from synthcity.plugins.core.models.syn_seq.methods.logreg import syn_logreg
from synthcity.plugins.core.models.syn_seq.methods.norm import syn_norm, syn_lognorm
from synthcity.plugins.core.models.syn_seq.methods.pmm import syn_pmm
from synthcity.plugins.core.models.syn_seq.methods.polyreg import syn_polyreg
from synthcity.plugins.core.models.syn_seq.methods.rf import syn_rf
from synthcity.plugins.core.models.syn_seq.methods.misc import syn_random, syn_swr


def _to_synseq_constraints(
    constraint_input: Union[None, Dict[str, List[Any]], Constraints]
) -> Optional[SynSeqConstraints]:
    """
    Converts user-supplied constraints (dict or Constraints object) 
    into a SynSeqConstraints object, so we can do direct substitution or row filtering.
    """
    if constraint_input is None:
        return None

    if isinstance(constraint_input, Constraints):
        if isinstance(constraint_input, SynSeqConstraints):
            return constraint_input
        else:
            return SynSeqConstraints(rules=constraint_input.rules)

    if isinstance(constraint_input, dict):
        rules_list = []
        for col, rule_list in constraint_input.items():
            if not isinstance(rule_list, list) or len(rule_list) < 2:
                print(f"[WARNING] Malformed constraint for '{col}' => {rule_list}")
                continue
            op = rule_list[0]
            val = rule_list[1]
            rules_list.append((col, op, val))
        return SynSeqConstraints(rules=rules_list)

    raise ValueError(f"Unsupported constraint type: {type(constraint_input)}")


class Syn_Seq:
    """
    A 'synthpop'-like sequential aggregator for synthetic data generation.
    
    Overview of the process:
      - We accept a Syn_SeqDataLoader, but we do NOT automatically rely on the 
        numeric splitting from the loader itself. Instead, we explicitly call
        'encode(...)' here using the same Syn_SeqEncoder. 
      - This means, at .fit() time, we transform the user’s data => an encoded DataFrame
        with any numeric columns possibly split into "col" + "col_cat." We then fit
        each column method (cart/pmm/etc.) on that encoded data.
      - At .generate() time, we produce synthetic data in that encoded space, then 
        call 'decode(...)' to revert to the user’s original columns.
      - If constraints are provided, we either do repeated tries ('strict=True') 
        or a single pass filtering approach.

    NOTE: We are NOT modifying the DataLoader or the Syn_SeqEncoder code itself. 
          We only use them as intended: 'encode' for splitting before .fit, 
          then 'decode' after generating the final data.
    """

    def __init__(
        self,
        random_state: int = 0,
        default_first_method: str = "SWR",
        default_other_method: str = "CART",
        strict: bool = True,
        sampling_patience: int = 500,
        seq_id_col: str = "seq_id",
        seq_time_col: str = "seq_time_id",
        **kwargs: Any
    ):
        """
        Args:
            random_state: for reproducibility
            default_first_method: fallback if the user does not specify method for col 0
            default_other_method: fallback for subsequent columns
            strict: if True => repeated tries to meet constraints
            sampling_patience: max tries in strict mode
            seq_id_col, seq_time_col: used by constraints if needed
            **kwargs: aggregator-level arguments (unused here)
        """
        self.random_state = random_state
        self.default_first_method = default_first_method
        self.default_other_method = default_other_method
        self.strict = strict
        self.sampling_patience = sampling_patience
        self.seq_id_col = seq_id_col
        self.seq_time_col = seq_time_col

        # Store per-column model info: method, predictor columns, fit_info, etc.
        self._column_models: Dict[str, Dict[str, Any]] = {}

        # The user can pass a list of methods or a variable_selection dict
        self.method_list: List[str] = []
        self.variable_selection: Dict[str, List[str]] = {}

        # For encode/decode usage
        self._encoders: Dict[str, Any] = {}

        self._model_trained = False

    # ----------------------------------------------------------------
    # fit(...)
    # ----------------------------------------------------------------
    def fit(
        self,
        loader: Syn_SeqDataLoader,
        method: Optional[List[str]] = None,
        variable_selection: Optional[Dict[str, List[str]]] = None,
        *args: Any,
        **kwargs: Any
    ) -> "Syn_Seq":
        """
        1) We call loader.encode(...) => splitted/encoded data, plus the encoder dict.
        2) Build predictor matrix from user variable_selection or default sequential logic.
        3) For each (encoded) column => store a partial model fit.
        """
        if not isinstance(loader, Syn_SeqDataLoader):
            raise TypeError("Syn_Seq aggregator requires a Syn_SeqDataLoader.")

        # 1) encode => splitted/encoded data
        encoded_loader, self._encoders = loader.encode(encoders=None)
        df_encoded = encoded_loader.dataframe()
        if df_encoded.empty:
            raise ValueError("No data after encoding. Cannot train on empty DataFrame.")

        self.method_list = method or []
        self.variable_selection = variable_selection or {}

        col_list = list(df_encoded.columns)
        n_cols = len(col_list)

        # 2) expand user methods if needed
        final_methods = []
        for i, col in enumerate(col_list):
            if i < len(self.method_list):
                final_methods.append(self.method_list[i])
            else:
                fallback = (
                    self.default_first_method if i == 0 else self.default_other_method
                )
                final_methods.append(fallback)

        # Build a default predictor matrix
        vs_matrix = pd.DataFrame(0, index=col_list, columns=col_list)
        for i in range(n_cols):
            vs_matrix.iloc[i, :i] = 1

        # incorporate user variable_selection
        for target_col, pred_cols in self.variable_selection.items():
            if target_col in vs_matrix.index:
                vs_matrix.loc[target_col, :] = 0
                for pc in pred_cols:
                    if pc in vs_matrix.columns:
                        vs_matrix.loc[target_col, pc] = 1

        print("[INFO] aggregator: final method assignment:")
        for col, meth in zip(col_list, final_methods):
            print(f"   - {col} => {meth}")
        print("[INFO] aggregator: final variable_selection matrix:")
        print(vs_matrix)

        # 3) train each encoded column
        self._column_models.clear()
        for i, col in enumerate(col_list):
            chosen_method = final_methods[i]
            pred_mask = vs_matrix.loc[col] == 1
            preds = vs_matrix.columns[pred_mask].tolist()

            y = df_encoded[col].values
            X = df_encoded[preds].values if preds else np.zeros((len(y), 0))

            print(f"[INFO] Fitting encoded column '{col}' with method '{chosen_method}'...")
            self._column_models[col] = {
                "method": chosen_method,
                "predictors": preds,
                "fit_info": self._fit_single_column(y, X, chosen_method),
            }

        self._model_trained = True
        return self

    def _fit_single_column(
        self, y: np.ndarray, X: np.ndarray, method: str
    ) -> Dict[str, Any]:
        """
        For each column, store (obs_y, obs_X) and the method type. 
        Actual training might happen again at generation time or partial store here.
        """
        method_lower = method.strip().lower()

        if method_lower in {
            "cart", "ctree", "rf", "norm", "lognorm", 
            "pmm", "logreg", "polyreg",
        }:
            return {"type": method_lower, "obs_y": y, "obs_X": X}
        elif method_lower == "swr":
            return {"type": "swr", "obs_y": y}
        elif method_lower == "random":
            return {"type": "random", "obs_y": y}
        else:
            # fallback => random
            return {"type": "random", "obs_y": y}

    # ----------------------------------------------------------------
    # generate(...)
    # ----------------------------------------------------------------
    def generate(
        self,
        count: int,
        constraint: Union[None, Dict[str, List[Any]], Constraints] = None,
        *args: Any,
        **kwargs: Any
    ) -> Syn_SeqDataLoader:
        """
        1) Generate 'encoded' synthetic data column by column.
        2) If constraints => either repeated tries or single pass filter.
        3) Finally, 'decode' to revert to the original user columns 
           (merging splitted numeric columns, etc.).
        """
        if not self._model_trained:
            raise RuntimeError("Must fit Syn_Seq before calling generate().")

        # unify constraints as a SynSeqConstraints
        syn_constraints = _to_synseq_constraints(constraint)
        if syn_constraints:
            syn_constraints.seq_id_feature = self.seq_id_col
            syn_constraints.seq_time_id_feature = self.seq_time_col

        # Use strict or single-pass approach
        if self.strict and syn_constraints is not None:
            encoded_df = self._attempt_strict_generation(count, syn_constraints)
        else:
            encoded_df = self._generate_once(count)
            if syn_constraints:
                encoded_df = self._apply_constraint_corrections(encoded_df, syn_constraints)
                encoded_df = syn_constraints.match(encoded_df)

        # Now decode to revert from splitted columns => original user columns
        # Wrap 'encoded_df' in a minimal loader so we can call decode(...)
        temp_loader = Syn_SeqDataLoader(data=encoded_df, syn_order=list(encoded_df.columns))
        decoded_loader = temp_loader.decode(self._encoders)

        # Return the final (decoded) data loader
        return decoded_loader

    def _generate_once(self, count: int) -> pd.DataFrame:
        """
        Single pass ignoring constraints. 
        Produces data in the encoded space (splitted columns).
        """
        col_list = list(self._column_models.keys())
        syn_df = pd.DataFrame(index=range(count))

        for col in col_list:
            info = self._column_models[col]
            method = info["method"]
            preds = info["predictors"]
            fit_data = info["fit_info"]

            Xp = syn_df[preds].values if preds else np.zeros((count, 0))
            print(f"[INFO] Generating encoded column '{col}' with method '{method}'...")

            new_vals = self._generate_single_column(method, fit_data, Xp, count)
            syn_df[col] = new_vals

        return syn_df

    def _generate_single_column(
        self,
        method: str,
        fit_model: Dict[str, Any],
        Xp: np.ndarray,
        count: int
    ) -> pd.Series:
        """
        Calls the appropriate method for generating that encoded column's data.
        """
        method_lower = method.strip().lower()

        y_obs = fit_model.get("obs_y", None)
        X_obs = fit_model.get("obs_X", None)

        if method_lower == "cart":
            res = syn_cart(y=y_obs, X=X_obs, Xp=Xp, random_state=self.random_state)
            return pd.Series(res["res"])
        elif method_lower == "ctree":
            res = syn_ctree(y=y_obs, X=X_obs, Xp=Xp, random_state=self.random_state)
            return pd.Series(res["res"])
        elif method_lower == "rf":
            res = syn_rf(y=y_obs, X=X_obs, Xp=Xp, random_state=self.random_state)
            return pd.Series(res["res"])
        elif method_lower == "norm":
            res = syn_norm(y=y_obs, X=X_obs, Xp=Xp, random_state=self.random_state)
            return pd.Series(res["res"])
        elif method_lower == "lognorm":
            res = syn_lognorm(y=y_obs, X=X_obs, Xp=Xp, random_state=self.random_state)
            return pd.Series(res["res"])
        elif method_lower == "pmm":
            res = syn_pmm(y=y_obs, X=X_obs, Xp=Xp, random_state=self.random_state)
            return pd.Series(res["res"])
        elif method_lower == "logreg":
            res = syn_logreg(y=y_obs, X=X_obs, Xp=Xp, random_state=self.random_state)
            return pd.Series(res["res"])
        elif method_lower == "polyreg":
            res = syn_polyreg(y=y_obs, X=X_obs, Xp=Xp, random_state=self.random_state)
            return pd.Series(res["res"])
        elif method_lower == "swr":
            res = syn_swr(y=y_obs, X=None, Xp=np.zeros((count, 1)), random_state=self.random_state)
            return pd.Series(res["res"])
        elif method_lower == "random":
            res = syn_random(y=y_obs, X=None, Xp=np.zeros((count, 1)), random_state=self.random_state)
            return pd.Series(res["res"])
        else:
            # fallback => random
            fallback = syn_random(y=y_obs, X=None, Xp=np.zeros((count, 1)), random_state=self.random_state)
            return pd.Series(fallback["res"])

    # ----------------------------------------------------------------
    # strict approach
    # ----------------------------------------------------------------
    def _attempt_strict_generation(
        self,
        count: int,
        syn_constraints: SynSeqConstraints
    ) -> pd.DataFrame:
        """
        If 'strict' => we generate repeatedly until constraints are satisfied 
        or we exhaust the sampling_patience. 
        We keep accumulating unique rows in the encoded space.
        """
        result_df = pd.DataFrame()
        tries = 0
        while len(result_df) < count and tries < self.sampling_patience:
            tries += 1
            chunk = self._generate_once(count)

            chunk = self._apply_constraint_corrections(chunk, syn_constraints)
            chunk = syn_constraints.match(chunk)
            chunk = chunk.drop_duplicates()

            result_df = pd.concat([result_df, chunk], ignore_index=True)

        return result_df.head(count)

    def _apply_constraint_corrections(
        self,
        df: pd.DataFrame,
        syn_constraints: SynSeqConstraints
    ) -> pd.DataFrame:
        """
        For '=' constraints, do direct substitution first. 
        For other constraints => let match(...) do filtering.
        """
        new_df = df.copy()
        for (feature, op, val) in syn_constraints.rules:
            if op in ["=", "=="]:
                new_df = syn_constraints._correct(new_df, feature, op, val)
        return new_df
