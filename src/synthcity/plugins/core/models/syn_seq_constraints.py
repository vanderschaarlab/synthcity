# File: syn_seq_constraints.py

from typing import Any, List, Dict, Tuple, Union
import pandas as pd
import numpy as np

# We import the base Constraints to inherit from
from synthcity.plugins.core.constraints import Constraints

# 
# A single sub-rule is (feature, op, value).
# For "chained" logic, we might interpret:
#    "For rows that pass all sub-rules (1..k-1), apply sub-rule k as a filter or correction."
#
# Example constraint input:
#   {
#       "N1": [
#           ("C1", "in", ["AAA","BBB"]),
#           ("N1", ">", 125)
#       ]
#   }
# means:
#  1) If a row passes (C1 in [AAA,BBB]), 
#  2) Then also enforce (N1 > 125) on that row.
#
# If the first sub-rule is not satisfied, the second doesn't apply to that row.
# If the first sub-rule is satisfied but not the second => row fails entirely.
#

class SynSeqConstraints(Constraints):
    """
    An extension that supports "chained" constraints for sequential logic:
      - If sub-rule 1 is satisfied => we must also pass sub-rule 2, and so on.
      - 'chained_rules' can be a dict of { targetCol : [ (feature, op, val), (feature, op, val), ... ] }.
        Example:
          {
            "N1": [
                ("C1", "in", ["AAA","BBB"]),
                ("N1", ">", 125)
            ]
          }
        read as: "If (C1 in [AAA,BBB]) => enforce (N1 > 125)".
      
      Each sub-rule uses the same _eval logic for <, <=, >, >=, ==, in, etc.
      If a sub-rule fails => that entire row fails (filtered out) if using .match(), 
      or is corrected if possible (.correct).
    """

    def __init__(
        self,
        # You can still pass standard constraints as "rules",
        # or pass new "chained_rules" in dict format
        rules: List[Tuple[str, str, Any]] = None,
        chained_rules: Dict[str, List[Tuple[str, str, Any]]] = None,
        seq_id_feature: str = "seq_id",
        seq_time_id_feature: str = "seq_time_id",
        **kwargs: Any,
    ):
        # Let the base constructor handle standard 'rules'
        super().__init__(rules=rules if rules else [])
        self.seq_id_feature = seq_id_feature
        self.seq_time_id_feature = seq_time_id_feature

        # We'll store the "chained" sub-rules in a separate structure
        # keyed by "target column" or "some label"
        self.chained_rules = chained_rules if chained_rules else {}

    def match(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Overridden. We can do row-level or entire-sequence filtering. 
        If you still want entire-sequence filtering, you can define match_sequential() and call it below.
        For demonstration, let's do row-level filtering with chained sub-rules.
        """
        df_copy = X.copy()
        # first, apply base constraints (self.rules) at row-level:
        base_mask = super().filter(df_copy)
        df_filtered = df_copy[base_mask].copy()
        if df_filtered.empty:
            return df_filtered

        # next, apply "chained" constraints
        df_filtered = self._match_chained(df_filtered)
        return df_filtered

    def _match_chained(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        For each entry in self.chained_rules => interpret a *sequence* of sub-rules.
        If the row passes sub-rule[0], sub-rule[1], ... sub-rule[n-2], 
        then sub-rule[n-1] is also enforced. 
        If it fails at any step => the row is filtered out or "corrected" (depending on your logic).
        
        For simplicity below, we do "if sub-rule[i] is satisfied => proceed, else row is out."
        """
        df = X.copy()
        for target_key, rule_chain in self.chained_rules.items():
            # We interpret the chain in order
            # e.g. [("C1","in", [...]), ("N1",">",125)]
            # step i=0 => a filter => if row fails => out
            # step i=1 => a further filter => if row fails => out
            # etc.

            # We'll build a mask for these sub-rules
            keep_mask = pd.Series([True]*len(df), index=df.index)

            for (feature, op, operand) in rule_chain:
                cur_mask = self._eval(df, feature, op, operand)
                # only keep the rows that pass this sub-rule
                keep_mask = keep_mask & cur_mask

            # after we apply all sub-rules in the chain, 
            # rows that didn't pass => out
            df = df[keep_mask].copy()
            if df.empty:
                break

        return df

    def _eval(self, X: pd.DataFrame, feature: str, op: str, operand: Any) -> pd.Index:
        """
        If we want to also handle direct substitution '=' or '==' or to skip it, we can override _eval.
        If the user wants a different meaning for '=' in chain logic, we can do so.
        Otherwise, we rely on base Constraints._eval for <, <=, >, >=, ==, in, dtype, etc.
        """
        # If you want direct substitution for '=' or '==', do so in _correct or in some separate logic.
        if op in ["=", "=="]:
            # interpret as equality check
            return (X[feature] == operand) | X[feature].isna()
        else:
            # fallback to base method
            return super()._eval(X, feature, op, operand)

    def _correct(self, X: pd.DataFrame, feature: str, op: str, operand: Any) -> pd.DataFrame:
        """
        If user wants direct substitution for '=' => set that col's entire column to operand. 
        Or you can do row-level changes only for failing rows, etc.
        """
        if op in ["=", "=="]:
            X.loc[:, feature] = operand
            return X
        return super()._correct(X, feature, op, operand)

    # If you want entire-sequence logic, you'd define match_sequential below:

    # Example
    #     constraint = {
    #   "N1": [
    #     ["C1", "in", ["AAA","BBB"]],
    #     ["N1", ">", 125]
    #   ]
    # }

    # def match_sequential(self, X: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Example method that applies constraints at the group (sequence) level:
    #       - if a sub-rule fails for ANY row => remove entire sequence
    #       - or do direct substitution in entire sequence
    #     """
    #     df_copy = X.copy()
    #     base_mask = super().filter(df_copy)
    #     # group by seq_id
    #     grouped = df_copy.groupby(self.seq_id_feature)
    #
    #     keep_seq_ids = []
    #     for seq_id, group in grouped:
    #         # if all rows pass => keep entire sequence
    #         if base_mask[group.index].all():
    #             # next, check chained
    #             # for each sub-rule chain, we can test if group passes
    #             # if not => exclude entire seq
    #             # or we can do partial correction
    #             # ...
    #             keep_seq_ids.append(seq_id)
    #
    #     return df_copy[df_copy[self.seq_id_feature].isin(keep_seq_ids)]

