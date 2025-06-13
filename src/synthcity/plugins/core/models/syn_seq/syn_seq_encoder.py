# stdlib
from typing import Any, Dict, List, Optional

# third party
import pandas as pd
from pydantic import validate_arguments
from sklearn.base import BaseEstimator, TransformerMixin


class Syn_SeqEncoder(TransformerMixin, BaseEstimator):
    """
    A minimal encoder for syn_seq.
      - It assumes that preprocessing (date conversion, special value handling, etc.) has already been done in preprocess.py.
      - This encoder only sets up syn_order, method, and variable_selection.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        syn_order: Optional[List[str]] = None,
        method: Optional[Dict[str, str]] = None,
        variable_selection: Optional[Dict[str, List[str]]] = None,
        default_method: str = "cart",
    ) -> None:
        """
        Args:
            syn_order: User-specified column order. If empty, X.columns will be used in the prepare step.
            method: A dictionary mapping columns to methods (e.g., {"col": "rf", ...}).
            variable_selection: A dictionary mapping each target column to a list of predictors.
            default_method: The default method to use when none is specified (e.g., "cart").
        """
        self.syn_order = syn_order or []
        self.method = method or {}
        self.variable_selection_ = variable_selection or {}
        self.default_method = default_method

        self.col_map: Dict[str, Dict[str, Any]] = {}

    def prepare(self, X: pd.DataFrame) -> "Syn_SeqEncoder":
        """
        Prepares the encoder using the reference DataFrame X:
          1) Determines the syn_order.
          2) Assigns a method to each column.
          3) Sets up variable selection.
        Returns:
            The encoder instance.
        """
        self._set_syn_order(X)
        self._assign_method_to_cols()
        self._assign_variable_selection()
        return self

    def _set_syn_order(self, X: pd.DataFrame) -> None:
        """
        If syn_order is empty, use X.columns;
        Otherwise, filter out columns not present in X.
        """
        if not self.syn_order:
            self.syn_order = list(X.columns)
        else:
            self.syn_order = [c for c in self.syn_order if c in X.columns]

    def _assign_method_to_cols(self) -> None:
        """
        For the first column, if the user has not specified a method, use "swr".
        For the remaining columns, use the user-specified method if provided, otherwise use default_method.
        The chosen method is stored in col_map.
        """
        self.col_map.clear()
        for i, col in enumerate(self.syn_order):
            user_m = self.method.get(col)
            if i == 0:
                chosen = user_m if user_m else "swr"
            else:
                chosen = user_m if user_m else self.default_method

            self.col_map[col] = {"method": chosen}

    def _assign_variable_selection(self) -> None:
        """
        For columns where the user did not specify variable_selection,
        assign all columns preceding the current column as predictors.
        """
        for i, col in enumerate(self.syn_order):
            if col not in self.variable_selection_:
                self.variable_selection_[col] = self.syn_order[:i]

    def get_info(self) -> Dict[str, Any]:
        """
        Returns the encoder's information including syn_order, method, and variable_selection.
        """
        method_map = {}
        for col in self.col_map:
            method_map[col] = self.col_map[col]["method"]

        return {
            "syn_order": self.syn_order,
            "method": method_map,
            "variable_selection": self.variable_selection_,
        }
