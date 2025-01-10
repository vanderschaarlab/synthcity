# File: plugins/syn_seq/syn_seq_constraints.py

"""
A specialized Constraints class for SynSeq scenarios, 
where we must handle the sequential data format (seq_id_feature, seq_time_id_feature, etc.),
plus direct substitution for "=" constraints in the new plugin logic.
"""

from typing import Any, List, Tuple, Union
import pandas as pd
import numpy as np

# synthcity absolute
from synthcity.plugins.core.constraints import Constraints


class SynSeqConstraints(Constraints):
    """
    An extension of the base Constraints class to handle sequential data columns 
    AND direct substitution for '=' or '==' constraints if desired.

    Key differences from base:
      - We interpret '=' or '==' constraints as direct substitution. 
      - For row-level checks, we still do "match" or "filter" to see if row passes 
        constraints. For '='/'==' we treat it like '==' under the hood. 
      - If you want the entire sequence removed if any row fails constraints, 
        we have the `match_sequential` logic.
    """

    def __init__(
        self,
        rules: List[Tuple[str, str, Any]] = None,
        seq_id_feature: str = "seq_id",
        seq_time_id_feature: str = "seq_time_id",
        **kwargs: Any,
    ):
        """
        Args:
            rules: a list of (feature, op, threshold) constraints
                   e.g. [("N1", "=", 999), ("C1", "in", ["AAA","BBB"]), ...]
            seq_id_feature: name of the sequential ID column
            seq_time_id_feature: name of the time index column
            **kwargs: forwarded to the base class constructor.
        """
        super().__init__(rules=rules if rules else [], **kwargs)
        self.seq_id_feature = seq_id_feature
        self.seq_time_id_feature = seq_time_id_feature

    # ------------------------------------------------------------
    # Override base `_eval` so that '=' or '==' is treated 
    # as an equality check
    # ------------------------------------------------------------
    def _eval(self, X: pd.DataFrame, feature: str, op: str, operand: Any) -> pd.Index:
        """
        Evaluate row-by-row which rows pass a given (feature, op, operand).
        For '=' or '==' we do an equality check (like the base version).
        """
        if op in ["=", "=="]:
            # treat as '=='
            return (X[feature] == operand) | X[feature].isna()
        else:
            # fallback to the base constraints for <, >, in, etc.
            return super()._eval(X, feature, op, operand)

    # ------------------------------------------------------------
    # Override base `_correct` so that '=' or '==' triggers 
    # direct substitution
    # ------------------------------------------------------------
    def _correct(
        self, X: pd.DataFrame, feature: str, op: str, operand: Any
    ) -> pd.DataFrame:
        """
        If we want to fix/modify values in X that do not pass constraints, 
        we override this. 
        For '=' or '==' => set the entire column to 'operand'.
        For other ops => fallback to parent logic
        """
        if op in ["=", "=="]:
            # direct assignment
            X.loc[:, feature] = operand
            return X
        else:
            return super()._correct(X, feature, op, operand)

    # ------------------------------------------------------------
    # Optionally override match() to preserve entire sequences
    # if any row fails
    # ------------------------------------------------------------
    def match_sequential(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Example method that applies constraints at the group (sequence) level.
        If any row in a sequence fails => entire sequence is removed.
        """
        df_copy = X.copy()

        # base 'filter' gives us T/F for each row
        valid_mask = super().filter(df_copy)  # boolean per row
        grouped = df_copy.groupby(self.seq_id_feature)
        keep_seq_ids = []
        for seq_id, group in grouped:
            # if all rows in that group pass => keep entire sequence
            if valid_mask[group.index].all():
                keep_seq_ids.append(seq_id)

        return df_copy[df_copy[self.seq_id_feature].isin(keep_seq_ids)]

    def match(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        We can either do row-level filtering or entire-sequence filtering.
        For demonstration, let's do entire-sequence approach by default.
        If you prefer row-level, you can do `return super().match(X)` here.
        """
        return self.match_sequential(X)
