# synthcity/plugins/syn_seq/syn_seq_constraints.py

"""
A specialized Constraints class for SynSeq scenarios, where we must handle
the sequential data format (seq_id_feature, seq_time_id_feature, etc.),
on top of the existing constraint logic in synthcity.
"""

from typing import Any, List, Tuple, Union
import pandas as pd
import numpy as np

# synthcity absolute
from synthcity.plugins.core.constraints import Constraints


class SynSeqConstraints(Constraints):
    """
    An extension of the base Constraints class to handle sequential data columns.

    In syn_seq, we often have:
        - a "seq_id_feature" (e.g., 'seq_id'), which identifies each subject/row-group
        - a "seq_time_id_feature" (e.g., 'seq_time_id'), which identifies time steps
        - specialized columns like 'seq_static_*' or 'seq_temporal_*' or 'seq_out_*'.

    This class can incorporate additional logic or overrides for these sequential
    columns if needed, e.g. ignoring constraints on the 'seq_time_id' or
    systematically handling them differently.

    If no specialized logic is needed beyond the base constraints, this class
    simply inherits and can be used interchangeably with the base Constraints.
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
            rules: same as the base Constraints, a list of (feature, op, threshold) constraints
            seq_id_feature: name of the sequential ID column
            seq_time_id_feature: name of the time index column
            **kwargs: forwarded to the base class constructor.
        """
        super().__init__(rules=rules if rules else [], **kwargs)
        self.seq_id_feature = seq_id_feature
        self.seq_time_id_feature = seq_time_id_feature

    def match_sequential(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Example of a custom method that might apply constraints
        but preserve the sequential grouping logic. Currently, we
        just reuse the base `match` method. But if you need to
        do something special for each sequence ID, do it here.

        Returns:
            Filtered DataFrame that meets the constraints
            across each seq_id group.
        """
        # Simple approach: just call base match
        # If you want to do something special like
        # preserve entire sequence if ANY row in it fails,
        # or if ALL rows must pass, etc., you can do it here.
        df_copy = X.copy()

        # EXAMPLE: If you wanted to remove entire sequences if ANY row fails:
        # We can group by seq_id_feature, check if all pass constraints,
        # and keep or remove them.
        # NOTE: The base 'match' method returns the rows that individually pass constraints.
        # Suppose we want the entire sequence if *all rows in that sequence pass*.
        valid_mask = super().filter(df_copy)  # True/False per row
        # Example logic: if ANY row is invalid => remove entire sequence
        grouped = df_copy.groupby(self.seq_id_feature)
        keep_seq_ids = []
        for seq_id, group in grouped:
            # if all rows are valid => keep that entire sequence
            if valid_mask[group.index].all():
                keep_seq_ids.append(seq_id)

        # now filter entire dataframe to only keep seq_id in keep_seq_ids
        final_df = df_copy[df_copy[self.seq_id_feature].isin(keep_seq_ids)]
        return final_df

    def match(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Override base method to incorporate the custom sequential logic if desired.
        If you do not need special logic, you can just call the base class method.
        """
        # For demonstration: let's show how you'd use match_sequential
        # to remove entire sequences if any row fails constraints.
        # If you prefer row-level filtering, simply do: return super().match(X)
        return self.match_sequential(X)

    # If you need to ignore constraints on time ID or seq ID columns, you can override:
    # e.g. in _eval or _correct, skip if feature == self.seq_id_feature or ...
    # But here we keep the base logic for demonstration.
