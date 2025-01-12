# File: syn_seq.py
#
# A self-contained aggregator for the sequential column-by-column approach,
# with no separate _fit or _generate. We unify logic into fit(...) and generate(...).
# We now incorporate the `SynSeqConstraints` (or base Constraints) usage for
# both direct substitution AND row/sequence filtering.

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from random import sample

# local references
from src.synthcity.plugins.core.dataloader import Syn_SeqDataLoader
from src.synthcity.plugins.core.constraints import Constraints
from src.synthcity.plugins.core.models.syn_seq_constraints import SynSeqConstraints


def _to_synseq_constraints(
    constraint_input: Union[None, Dict[str, List[Any]], Constraints]
) -> Optional[SynSeqConstraints]:
    """
    A helper that converts user-supplied constraints (dict or Constraints)
    into a SynSeqConstraints object.
    
    Example dictionary format:
      {
         "N1": ["=", 999],
         "C2": ["in", ["A","B"]]
      }
    => [("N1","=",999),("C2","in",["A","B"])]

    If the user already has a SynSeqConstraints or base Constraints, we wrap or copy.
    """
    if constraint_input is None:
        return None
    
    if isinstance(constraint_input, Constraints):
        # Already a (Syn)Constraints?
        if isinstance(constraint_input, SynSeqConstraints):
            return constraint_input
        else:
            # Copy its rules into a SynSeqConstraints
            return SynSeqConstraints(rules=constraint_input.rules)
    
    if isinstance(constraint_input, dict):
        # Simple parse: each key => (op, val)
        # e.g. "col" : ["=", 999]
        # if len(...) < 2 => skip
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
    The aggregator model for a sequential (column-by-column) synthetic data approach.

    - Provides public .fit(...) and .generate(...) (no separate _fit/_generate).
    - Maintains per-column model info, variable selection, constraints, etc.
    - Integrates `SynSeqConstraints` for direct substitution ('=') and row/sequence filtering.

    Usage:
        aggregator = Syn_Seq(...)
        aggregator.fit(loader, method=[...], variable_selection={...})
        syn_data = aggregator.generate(count=..., constraint={...})
    """

    def __init__(
        self,
        random_state: int = 0,
        default_first_method: str = "SWR",
        default_other_method: str = "CART",
        strict: bool = True,
        sampling_patience: int = 500,
        seq_id_col: str = "seq_id",   # if you want sequence-level constraints
        seq_time_col: str = "seq_time_id",
        **kwargs: Any
    ):
        """
        Args:
            random_state: for reproducibility.
            default_first_method: fallback for the first column if not user-specified.
            default_other_method: fallback for subsequent columns if not user-specified.
            strict: if True, constraints are strictly enforced with repeated tries.
            sampling_patience: how many times we attempt new draws if constraints fail.
            seq_id_col: for sequence-level constraints in SynSeqConstraints.
            seq_time_col: time index column, if needed.
            **kwargs: aggregator-level arguments (unused here).
        """
        self.random_state = random_state
        self.default_first_method = default_first_method
        self.default_other_method = default_other_method
        self.strict = strict
        self.sampling_patience = sampling_patience

        # Per-column model info
        self._column_models: Dict[str, Any] = {}

        # For user-supplied or fallback
        self.method_list: List[str] = []
        self.variable_selection: Dict[str, List[str]] = {}

        # If used by constraints
        self.seq_id_col = seq_id_col
        self.seq_time_col = seq_time_col

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
        Fit a column-by-column model.
         1) Expand user method array if needed
         2) Build variable-selection matrix
         3) Train minimal model for each column
        """
        df = loader.dataframe()
        if df.empty:
            raise ValueError("No data in Syn_SeqDataLoader for training.")

        self.method_list = method or []
        self.variable_selection = variable_selection or {}

        col_list = list(df.columns)
        n_cols = len(col_list)

        # 1) Expand methods if needed
        final_methods = []
        for i, col in enumerate(col_list):
            if i < len(self.method_list):
                final_methods.append(self.method_list[i])
            else:
                # fallback
                fallback = self.default_first_method if i == 0 else self.default_other_method
                final_methods.append(fallback)

        # 2) Build variable_selection matrix
        vs_matrix = pd.DataFrame(0, index=col_list, columns=col_list)
        for i in range(n_cols):
            vs_matrix.iloc[i, :i] = 1

        # incorporate user-specified variable_selection
        for target_col, pred_cols in self.variable_selection.items():
            if target_col in vs_matrix.index:
                vs_matrix.loc[target_col, :] = 0
                for pc in pred_cols:
                    if pc in vs_matrix.columns:
                        vs_matrix.loc[target_col, pc] = 1

        print("[INFO] aggregator: final method assignment:")
        for col, m in zip(col_list, final_methods):
            print(f"  {col} => {m}")
        print("[INFO] aggregator: final variable_selection matrix:")
        print(vs_matrix)

        # 3) Train a minimal "model" for each column
        self._column_models.clear()
        for i, col in enumerate(col_list):
            chosen_method = final_methods[i]
            preds = vs_matrix.columns[(vs_matrix.loc[col] == 1)].tolist()
            model_info = self._train_column_model(df, target_col=col, predictor_cols=preds, method=chosen_method)
            self._column_models[col] = {
                "method": chosen_method,
                "predictors": preds,
                "model": model_info,
            }

        self._model_trained = True
        return self

    # ----------------------------------------------------------------
    # generate(...)
    # ----------------------------------------------------------------
    def generate(
        self,
        count: int = 10,
        constraint: Union[None, Dict[str, List[Any]], Constraints] = None,
        *args: Any,
        **kwargs: Any
    ) -> Syn_SeqDataLoader:
        """
        Generate synthetic data row-by-row.

        Steps:
          1) Convert `constraint` to SynSeqConstraints if needed
          2) If strict => repeated tries
             else => single pass + constraints
          3) Return a new Syn_SeqDataLoader
        """
        if not self._model_trained:
            raise RuntimeError("fit() must be called before generate().")

        # 1) unify constraints
        syn_constraints = _to_synseq_constraints(constraint)
        if syn_constraints is not None:
            syn_constraints.seq_id_feature = self.seq_id_col
            syn_constraints.seq_time_id_feature = self.seq_time_col

        # 2) Strict => repeated tries
        if syn_constraints and self.strict:
            syn_df = self._attempt_strict_generation(count, syn_constraints)
        else:
            # single pass
            syn_df = self._generate_once(count)
            # direct substitution + match if we have constraints
            if syn_constraints:
                syn_df = self._apply_synseq_corrections(syn_df, syn_constraints)
                syn_df = syn_constraints.match(syn_df)

        # 3) wrap up
        return Syn_SeqDataLoader(data=syn_df, syn_order=list(syn_df.columns))

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def _train_column_model(
        self,
        df: pd.DataFrame,
        target_col: str,
        predictor_cols: List[str],
        method: str,
    ) -> dict:
        """
        Minimal training logic; store raw data for now. 
        Real use would implement CART, pmm, etc.
        """
        model_info = {
            "predictors": predictor_cols,
            "target_data": df[target_col].values,
            "method": method,
        }
        return model_info

    def _generate_for_column(
        self,
        count: int,
        col: str,
        method: str,
        predictor_cols: List[str],
        model_obj: dict,
        partial_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Column-level sampling approach:
          - "swr" => sample without replacement
          - "cart", "pmm" => placeholder => random from real
          - fallback => random from real
        """
        real_data = model_obj["target_data"]
        rng = np.random.default_rng(self.random_state + hash(col) % 999999)

        if method.lower() == "swr":
            n_real = len(real_data)
            if count <= n_real:
                picks = sample(list(real_data), count)
            else:
                picks = list(real_data)
                overshoot = count - n_real
                picks += sample(list(real_data), overshoot)
            return pd.Series(picks)

        elif method.lower() in ["cart", "pmm"]:
            # placeholder => random
            picks = rng.choice(real_data, size=count, replace=True)
            return pd.Series(picks)

        else:
            # fallback => random
            picks = rng.choice(real_data, size=count, replace=True)
            return pd.Series(picks)

    def _generate_once(self, count: int) -> pd.DataFrame:
        """
        Single pass ignoring constraints.
        """
        col_list = list(self._column_models.keys())
        syn_df = pd.DataFrame(index=range(count))
        for col in col_list:
            info = self._column_models[col]
            method = info["method"]
            preds = info["predictors"]
            model_obj = info["model"]

            new_vals = self._generate_for_column(
                count, col, method, preds, model_obj, partial_df=syn_df
            )
            syn_df[col] = new_vals

        return syn_df

    def _attempt_strict_generation(
        self,
        count: int,
        syn_constraints: SynSeqConstraints
    ) -> pd.DataFrame:
        """
        Repeated tries if strict => generate, correct '=' constraints, match => 
        accumulate until we have `count` rows or out of patience.
        """
        result_df = pd.DataFrame()
        tries = 0
        while len(result_df) < count and tries < self.sampling_patience:
            tries += 1
            chunk = self._generate_once(count)
            chunk = self._apply_synseq_corrections(chunk, syn_constraints)
            chunk = syn_constraints.match(chunk)
            chunk = chunk.drop_duplicates()

            result_df = pd.concat([result_df, chunk], ignore_index=True)

        return result_df.head(count)

    def _apply_synseq_corrections(
        self,
        df: pd.DataFrame,
        syn_constraints: SynSeqConstraints
    ) -> pd.DataFrame:
        """
        For each (feature, op, val) in constraints, if op in ['=','=='],
        do direct substitution. For other ops => do nothing here.
        """
        new_df = df.copy()
        for (feature, op, val) in syn_constraints.rules:
            if op in ["=", "=="]:
                new_df = syn_constraints._correct(new_df, feature, op, val)
        return new_df
