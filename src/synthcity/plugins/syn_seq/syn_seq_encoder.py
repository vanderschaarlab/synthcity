# synthcity/plugins/syn_seq/syn_seq_encoder.py

from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator


class Syn_SeqEncoder(TransformerMixin, BaseEstimator):
    """
    Syn_SeqEncoder handles preprocessing and postprocessing tasks using fit/transform pattern.
    """

    def __init__(
        self,
        columns_special_values: Optional[Dict[str, Any]] = None,
        target_order: Optional[List[str]] = None,
        unique_value_threshold: int = 20,
    ) -> None:
        self.columns_special_values = columns_special_values or {}
        self.target_order = target_order or []
        self.unique_value_threshold = unique_value_threshold

        self.categorical_info_ = {}
        self.numeric_info_ = {}
        self.column_order_ = None
        self.method_assignments = {}
        self.prediction_matrix = None

    def fit(self, X: pd.DataFrame, y=None) -> "Syn_SeqEncoder":
        """Collect transform info needed at training time."""
        X = X.copy()

        # 1. save column order
        self.column_order_ = (
            self.target_order if self.target_order else X.columns.tolist()
        )

        # 2. decide col type by unique-value threshold
        for col in X.columns:
            unique_values = X[col].nunique()
            if unique_values > self.unique_value_threshold:
                # treat as numeric
                self.numeric_info_[col] = {"dtype": X[col].dtype}
            else:
                self.categorical_info_[col] = {"dtype": X[col].dtype}

        # 3. Example: if freq > 0.9 for some value, treat it as special
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            high_freq_vals = freq[freq > 0.9].index.tolist()
            if high_freq_vals:
                self.columns_special_values[col] = (
                    self.columns_special_values.get(col, []) + high_freq_vals
                )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Use fitted info to transform data."""
        X = X.copy()
        X = self._reorder_columns(X)
        X = self._categorize_numeric(X)
        X = self._update_column_types(X)
        X = self._assign_methods(X)
        X = self._generate_prediction_matrix(X)
        return X

    def _reorder_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.column_order_:
            return X[self.column_order_]
        return X

    def _categorize_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.numeric_info_:
            if col in X.columns:
                new_col = f"{col}_cat"
                special_values = self.columns_special_values.get(col, [])
                # example
                X[new_col] = X[col].apply(
                    lambda x: x if x in special_values or pd.isna(x) else -777
                )
                X[new_col] = X[new_col].fillna(-9999)
                X[col] = X[col].apply(
                    lambda x: x if x not in special_values and not pd.isna(x) else pd.NA
                )
                X[new_col] = X[new_col].astype("category")
        return X

    def _update_column_types(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in X.columns:
            if col in self.numeric_info_:
                X[col] = X[col].astype(self.numeric_info_[col]["dtype"])
            elif col in self.categorical_info_:
                X[col] = X[col].astype(self.categorical_info_[col]["dtype"])
        return X

    def _assign_methods(self, X: pd.DataFrame) -> pd.DataFrame:
        first_col = True
        for col in X.columns:
            if first_col:
                self.method_assignments[col] = "random_sampling"
                first_col = False
            else:
                self.method_assignments[col] = "CART"
        return X

    def _generate_prediction_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        self.prediction_matrix = pd.DataFrame(index=X.index, columns=X.columns)
        for col in X.columns:
            self.prediction_matrix[col] = 0
        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # restore special values
        for col, special_val in self.columns_special_values.items():
            if col in X.columns:
                X[col] = X[col].replace(pd.NA, special_val)

        # restore data types
        for col, info in self.categorical_info_.items():
            if col in X.columns:
                X[col] = X[col].astype(info["dtype"])

        for col, info in self.numeric_info_.items():
            if col in X.columns:
                X[col] = X[col].astype(info["dtype"])

        return X
