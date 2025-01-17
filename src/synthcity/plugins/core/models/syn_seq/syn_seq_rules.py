# File: syn_seq_rules.py

from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np

"""
We want to accept something like:

   rules = {
     "target": [
       ("bmi", ">", 0.15),
       ("target", ">", 0)
     ],
     "X3": [
       ("X1", "<", 0),
       ("X3", ">", 0)
     ]
   }

We interpret each key as "col to be tested/regenerated" => list of conditions
Each condition is (col_if, op, val_if).
We then do "if row satisfies (col_if op val_if), then row must also satisfy col->some constraints"
But you said you prefer to re-generate the values until conditions are met or until max iteration.
If it fails, we fallback to np.nan or some default.

Hence, the logic can be:
   For each row in generated_data,
       while not satisfied(rules for col) and iteration < MAX:
          y_syn[i, col] = regenerate(....)
       if still not satisfied => y_syn[i, col] = np.nan
"""


class Syn_SeqRules:
    def __init__(
        self,
        chained_rules: Optional[Dict[str, List[Tuple[str, str, Any]]]] = None,
        max_iterations: int = 20,
    ) -> None:
        """
        chained_rules example:
          {
             "X3":[("X1","<",0),("X3",">",0)],
             "target":[("bmi",">",0.15),("target",">",0)]
          }
        """
        self.rules = chained_rules if chained_rules else {}
        self.max_iterations = max_iterations

    def apply_rules(
        self,
        data: pd.DataFrame,
        col: str,
        generation_callback,
        preds_list: List[str],
        col_model: Any,
    ) -> pd.DataFrame:
        """
        For each row, if the rules are not satisfied, we re-generate the col's value (calling generation_callback).
        We do this up to self.max_iterations times. If it fails, set np.nan.
        generation_callback is something like generate_col(...) that returns a new value for that row.
        preds_list: the predictor columns for this col
        col_model: the fitted model for this col
        """
        if col not in self.rules:
            return data

        conditions = self.rules[col]  # list of (col_if, op, val)
        for i in range(len(data)):
            iteration_count = 0
            while iteration_count < self.max_iterations:
                if self._satisfied(data, i, conditions):
                    break
                # Re-generate that single value
                new_val = generation_callback(
                    col_model,
                    data.loc[[i], preds_list],
                )
                data.loc[i, col] = new_val
                iteration_count += 1
            if not self._satisfied(data, i, conditions):
                data.loc[i, col] = np.nan  # fallback

        return data

    def _satisfied(
        self, data: pd.DataFrame, row_index: int, conditions: List[Tuple[str, str, Any]]
    ) -> bool:
        """
        Check if the row data satisfies all conditions in conditions
        Each condition is (col_if, op, val)
        """
        row = data.iloc[row_index]
        for (cif, op, val) in conditions:
            if not self._op_check(row[cif], op, val):
                return False
        return True

    def _op_check(self, left, op, right):
        if pd.isna(left):
            return False
        if op == "<":
            return left < right
        elif op == "<=":
            return left <= right
        elif op == ">":
            return left > right
        elif op == ">=":
            return left >= right
        elif op in ["==", "="]:
            return left == right
        else:
            # fallback
            return False
