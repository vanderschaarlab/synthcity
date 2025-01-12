# File: syn_seq.py

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

# synergy imports
from synthcity.plugins.core.dataloader import Syn_SeqDataLoader
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.models.syn_seq.syn_seq_constraints import SynSeqConstraints

# 1) Now we import from the methods subpackage
from synthcity.plugins.core.models.syn_seq.methods.cart import syn_cart
from synthcity.plugins.core.models.syn_seq.methods.ctree import syn_ctree
from synthcity.plugins.core.models.syn_seq.methods.logreg import syn_logreg
from synthcity.plugins.core.models.syn_seq.methods.norm import syn_norm, syn_lognorm
from synthcity.plugins.core.models.syn_seq.methods.pmm import syn_pmm
from synthcity.plugins.core.models.syn_seq.methods.polyreg import syn_polyreg
from synthcity.plugins.core.models.syn_seq.methods.rf import syn_rf

# If you'd like all "misc" functions: syn_random, syn_swr, etc.
from synthcity.plugins.core.models.syn_seq.methods.misc import syn_random, syn_swr, syn_constant



def _to_synseq_constraints(
    constraint_input: Union[None, Dict[str, List[Any]], Constraints]
) -> Optional[SynSeqConstraints]:
    """
    A helper that converts user-supplied constraints (dict or Constraints)
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
    A 'synthpop'-like sequential aggregator:
      - Takes a Syn_SeqDataLoader (which has been encoded/split by Syn_SeqEncoder).
      - Column-by-column, we fit the chosen method (e.g., cart, pmm, logreg, ...).
      - Then we can generate new synthetic data in the same sequence.
      - We can optionally apply constraints strictly or in a single pass.
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
            default_first_method: fallback if user does not specify a method for column 0
            default_other_method: fallback for columns 1..n
            strict: if True => repeated tries to meet constraints
            sampling_patience: max attempts
            seq_id_col, seq_time_col: used by constraints if needed
            **kwargs: aggregator-level arguments (unused)
        """
        self.random_state = random_state
        self.default_first_method = default_first_method
        self.default_other_method = default_other_method
        self.strict = strict
        self.sampling_patience = sampling_patience
        self.seq_id_col = seq_id_col
        self.seq_time_col = seq_time_col

        # store the final model info for each column
        self._column_models: Dict[str, Dict[str, Any]] = {}

        # user-supplied
        self.method_list: List[str] = []
        self.variable_selection: Dict[str, List[str]] = {}

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
        Fit columns in the loader's order:
            1) If user gave fewer methods => expand with defaults
            2) Build a predictor matrix from variable_selection or prior columns
            3) For each column => gather (X, y), call the appropriate method => store model
        """
        df = loader.dataframe()
        if df.empty:
            raise ValueError("No data in Syn_SeqDataLoader for training.")

        # store user instructions
        self.method_list = method or []
        self.variable_selection = variable_selection or {}

        col_list = list(df.columns)
        n_cols = len(col_list)

        # 1) expand user methods
        final_methods = []
        for i, col in enumerate(col_list):
            if i < len(self.method_list):
                final_methods.append(self.method_list[i])
            else:
                fallback = self.default_first_method if i == 0 else self.default_other_method
                final_methods.append(fallback)

        # 2) build a predictor matrix (row=target col=all prior)
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

        print("[INFO] Starting fit: final method assignment:")
        for col, meth in zip(col_list, final_methods):
            print(f"   - {col} => {meth}")
        print("[INFO] variable_selection matrix:\n", vs_matrix)

        # 3) train each column model
        self._column_models.clear()

        for i, col in enumerate(col_list):
            chosen_method = final_methods[i]
            pred_mask = vs_matrix.loc[col] == 1
            preds = vs_matrix.columns[pred_mask].tolist()

            y = df[col].values
            X = df[preds].values if preds else np.zeros((len(y), 0))

            print(f"[INFO] Fitting column '{col}' using method '{chosen_method}'...")
            trained_model_info = self._fit_single_column(y, X, chosen_method)
            self._column_models[col] = {
                "method": chosen_method,
                "predictors": preds,
                "fit_info": trained_model_info,
            }

        self._model_trained = True
        return self

    def _fit_single_column(
        self, y: np.ndarray, X: np.ndarray, method: str
    ) -> Dict[str, Any]:
        """
        Actually train the method for that column:
          e.g. cart => might do partial fit or store y, X for generation
        """
        method_lower = method.strip().lower()
        rng_int = self.random_state  # or pass in a more advanced approach

        # Basic dispatch. In real usage, you'd unify code for numeric vs. cat, etc.
        if method_lower == "cart":
            # We don't have to generate new data now, but we can store the model
            # Usually syn_cart is used at generate time, but let's do a "partial" fit:
            # If you'd like, you can store X,y or an actual trained tree.
            # For actual training, we might do:
            #   # We'll pass Xp=some dummy, since syn_cart needs Xp for generation
            #   # But let's do it at generation time. Another approach is "store y, X" only
            return {
                "type": "cart",
                "obs_y": y,
                "obs_X": X,
            }
        elif method_lower == "pmm":
            # Similar approach
            return {
                "type": "pmm",
                "obs_y": y,
                "obs_X": X,
            }
        elif method_lower in ["logreg"]:
            return {
                "type": "logreg",
                "obs_y": y,
                "obs_X": X,
            }
        elif method_lower in ["polyreg"]:
            return {
                "type": "polyreg",
                "obs_y": y,
                "obs_X": X,
            }
        elif method_lower == "swr":
            return {
                "type": "swr",
                "obs_y": y,
            }
        elif method_lower == "random":
            return {
                "type": "random",
                "obs_y": y,
            }
        else:
            # fallback
            return {
                "type": "random",
                "obs_y": y,
            }

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
        Create synthetic data:
            1) Convert constraint => SynSeqConstraints
            2) If strict => repeated tries, else => single pass
            3) Return new Syn_SeqDataLoader
        """
        if not self._model_trained:
            raise RuntimeError("Must call fit(...) before generate(...).")

        # unify constraints
        syn_constraints = _to_synseq_constraints(constraint)
        if syn_constraints:
            syn_constraints.seq_id_feature = self.seq_id_col
            syn_constraints.seq_time_id_feature = self.seq_time_col

        if self.strict and syn_constraints:
            # repeated tries
            df_synthetic = self._attempt_strict_generation(count, syn_constraints)
        else:
            df_synthetic = self._generate_once(count)
            # apply constraints in single pass
            if syn_constraints:
                df_synthetic = self._apply_constraint_corrections(df_synthetic, syn_constraints)
                df_synthetic = syn_constraints.match(df_synthetic)

        # wrap in a new Syn_SeqDataLoader
        syn_loader = Syn_SeqDataLoader(data=df_synthetic, syn_order=list(df_synthetic.columns))
        return syn_loader

    def _generate_once(self, count: int) -> pd.DataFrame:
        """
        Single pass: produce new data row-by-row in the same column order used in fit.
        """
        col_list = list(self._column_models.keys())
        syn_df = pd.DataFrame(index=range(count))

        for col in col_list:
            info = self._column_models[col]
            method = info["method"]
            preds = info["predictors"]
            fit_data = info["fit_info"]

            # gather predictor columns from syn_df
            Xp = syn_df[preds].values if preds else np.zeros((count, 0))

            print(f"[INFO] Generating column '{col}' with method '{method}'...")

            new_vals = self._generate_single_column(
                method=method,
                fit_model=fit_data,
                Xp=Xp,
                count=count,
            )
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
        Dispatch based on method => call the actual generation function (cart, pmm, etc.)
        """
        method_lower = method.strip().lower()
        rng_int = self.random_state  # or incorporate the column index

        y_obs = fit_model.get("obs_y", None)
        X_obs = fit_model.get("obs_X", None)

        if method_lower == "cart":
            result = syn_cart(
                y=y_obs,
                X=X_obs,
                Xp=Xp,
                random_state=rng_int,
            )
            return pd.Series(result["res"])
        if method_lower == "ctree":
            result = syn_ctree(
                y=y_obs,
                X=X_obs,
                Xp=Xp,
                random_state=rng_int,
            )
            return pd.Series(result["res"])
        if method_lower == "rf":
            result = syn_rf(
                y=y_obs,
                X=X_obs,
                Xp=Xp,
                random_state=rng_int,
            )
            return pd.Series(result["res"])
        if method_lower == "norm":
            result = syn_norm(
                y=y_obs,
                X=X_obs,
                Xp=Xp,
                random_state=rng_int,
            )
            return pd.Series(result["res"])
        elif method_lower == "pmm":
            result = syn_pmm(
                y=y_obs,
                X=X_obs,
                Xp=Xp,
                random_state=rng_int,
            )
            return pd.Series(result["res"])
        elif method_lower == "logreg":
            result = syn_logreg(
                y=y_obs,
                X=X_obs,
                Xp=Xp,
                random_state=rng_int,
            )
            return pd.Series(result["res"])
        elif method_lower == "polyreg":
            result = syn_polyreg(
                y=y_obs,
                X=X_obs,
                Xp=Xp,
                random_state=rng_int,
            )
            return pd.Series(result["res"])
        elif method_lower == "swr":
            # Sample Without Replacement
            # We'll treat Xp only for length
            result = syn_swr(
                y=y_obs,
                X=None,
                Xp=np.zeros((count,1)),  # just to define size
                random_state=rng_int
            )
            return pd.Series(result["res"])
        elif method_lower == "random":
            # purely random from y_obs
            result = syn_random(
                y=y_obs,
                X=None,
                Xp=np.zeros((count,1)),
                random_state=rng_int,
            )
            return pd.Series(result["res"])
        else:
            # fallback => random
            result = syn_random(
                y=y_obs,
                X=None,
                Xp=np.zeros((count,1)),
                random_state=rng_int,
            )
            return pd.Series(result["res"])

    # ----------------------------------------------------------------
    # strict approach
    # ----------------------------------------------------------------
    def _attempt_strict_generation(
        self,
        count: int,
        syn_constraints: SynSeqConstraints
    ) -> pd.DataFrame:
        """
        Repeatedly generate chunks, correct '=' constraints, then match => keep unique rows => stop if we have enough or out of tries.
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
        For '=' constraints, do direct substitution; for others, let .match() do filtering.
        """
        new_df = df.copy()
        for (feature, op, val) in syn_constraints.rules:
            if op in ["=", "=="]:
                new_df = syn_constraints._correct(new_df, feature, op, val)
        return new_df
