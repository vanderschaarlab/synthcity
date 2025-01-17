# File: syn_seq_rules.py

from typing import List, Dict, Any
import pandas as pd

class Syn_SeqRules:
    """
    A minimal implementation that reads user-provided rules:
      e.g. rules = {
         "target":[
             ("bmi", ">", 0.15),
             ("target", ">", 0)
         ],
         "bp":[
             ("bp", "=", 0)
         ]
      }
    The expected usage is:
      - For each col being generated, we check if col is in the rules dict.
      - If it is, we attempt to filter rows that do not satisfy all sub-rules => re-generate.
    """

    def __init__(self, chained_rules: Dict[str, List[Any]], max_iter: int = 10):
        """
        chained_rules: A dictionary => { "colname": [ (col/feature, op, val), (col, op, val), ... ], ... }
        max_iter: maximum iteration for re-generation attempts.
        """
        self.chained_rules = chained_rules
        self.max_iter = max_iter

    def check_violations(self, df: pd.DataFrame, target_col: str) -> pd.Index:
        """
        Return the df index that violates the rules for `target_col`.
        We expect rules for `target_col` => [("bmi", ">", 0.15), ("target", ">", 0), ...]
        We'll interpret each sub-rule as "df[feature] op val must be True" for it to be valid.
        Return all row indices that fail at least one sub-rule => they violate.
        """
        if target_col not in self.chained_rules:
            return pd.Index([])  # no rules => no violations

        sub_rules = self.chained_rules[target_col]
        # We gather boolean mask for "valid rows" across all sub-rules
        mask_valid = pd.Series([True]*len(df), index=df.index)

        for (col_feat, operator, value) in sub_rules:
            # If col_feat not in df => skip or treat it as no constraint
            if col_feat not in df.columns:
                continue

            if operator == "=" or operator == "==":
                local_mask = (df[col_feat] == value) | df[col_feat].isna()
            elif operator == ">":
                local_mask = (df[col_feat] > value) | df[col_feat].isna()
            elif operator == ">=":
                local_mask = (df[col_feat] >= value) | df[col_feat].isna()
            elif operator == "<":
                local_mask = (df[col_feat] < value) | df[col_feat].isna()
            elif operator == "<=":
                local_mask = (df[col_feat] <= value) | df[col_feat].isna()
            else:
                # Could expand for "!=" or "in", etc. as needed
                local_mask = pd.Series([True]*len(df), index=df.index)

            mask_valid &= local_mask

        # Violations = ~mask_valid
        violating_index = df.index[~mask_valid]
        return violating_index
