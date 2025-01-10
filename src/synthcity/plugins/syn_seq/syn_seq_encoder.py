from typing import Optional, Dict, Any, List, Union
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class Syn_SeqEncoder(TransformerMixin, BaseEstimator):
    """
    Syn_SeqEncoder handles preprocessing and postprocessing tasks using fit/transform pattern,
    plus manages a 'variable_selection' matrix akin to a prediction matrix.
    """

    def __init__(
        self,
        columns_special_values: Optional[Dict[str, Any]] = None,
        syn_order: Optional[List[str]] = None,
        max_categories: int = 20,
        user_variable_selection: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Args:
            columns_special_values : {colName : [specialVals...]}
            syn_order : column order
            max_categories : threshold for deciding numeric vs categorical
            user_variable_selection : optional DataFrame(n x n) (row=target, col=predictor)
        """
        self.columns_special_values = columns_special_values or {}
        self.syn_order = syn_order or []
        self.max_categories = max_categories

        # user-provided or None
        self.user_variable_selection = user_variable_selection

        self.categorical_info_ = {}
        self.numeric_info_ = {}
        self.column_order_: List[str] = []
        self.method_assignments: Dict[str, str] = {}

        # main variable_selection (prediction matrix)
        self.variable_selection_: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y=None) -> "Syn_SeqEncoder":
        X = X.copy()

        self._detect_column_order(X)
        self._detect_col_types(X)
        self._detect_special_values(X)
        self._build_variable_selection_matrix(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Step 1: reorder columns
        X = self._reorder_columns(X)
        # Step 2: split numeric -> numeric + numeric_cat
        X = self._split_numeric_cols(X)
        # Step 3: update dtypes
        X = self._update_column_types(X)
        # Step 4: assign methods
        X = self._assign_methods(X)

        # Step 5: update variable_selection for newly created columns
        self._update_variable_selection_after_split(X)

        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1) restore special values
        for col, vals in self.columns_special_values.items():
            if col in X.columns:
                # if vals is list, .replace(pd.NA, vals) might be ambiguous
                # ideally, assume 1:1 mapping => e.g. "pd.NA -> singleValue"
                # or handle multiple?
                if isinstance(vals, list) and len(vals) == 1:
                    vals = vals[0]
                X[col] = X[col].replace(pd.NA, vals)

        # 2) restore dtype
        for col, info in self.categorical_info_.items():
            if col in X.columns:
                X[col] = X[col].astype(info["dtype"])
        for col, info in self.numeric_info_.items():
            if col in X.columns:
                X[col] = X[col].astype(info["dtype"])

        return X

    # -----------------------------------------------------
    # variable_selection building
    # -----------------------------------------------------
    def _build_variable_selection_matrix(self, X: pd.DataFrame) -> None:
        n = len(self.column_order_)
        df_cols = self.column_order_

        if self.user_variable_selection is not None:
            vs = self.user_variable_selection.copy()
            if vs.shape != (n, n):
                raise ValueError(f"user_variable_selection must be shape ({n},{n}), got {vs.shape}")
            # check row/col alignment
            if list(vs.index) != df_cols or list(vs.columns) != df_cols:
                raise ValueError("Mismatch in user_variable_selection index/columns vs syn_order.")
            self.variable_selection_ = vs
        else:
            # default: row=target col=predictor
            vs = pd.DataFrame(0, index=df_cols, columns=df_cols)
            # e.g. for i-th col => use all j < i as predictor
            for i in range(n):
                for j in range(i):
                    vs.iat[i, j] = 1
            self.variable_selection_ = vs

    def _update_variable_selection_after_split(self, X: pd.DataFrame) -> None:
        if self.variable_selection_ is None:
            return
        old_vs = self.variable_selection_
        old_rows = old_vs.index.tolist()
        old_cols = old_vs.columns.tolist()

        final_cols = list(X.columns)
        new_cols = [c for c in final_cols if c not in old_rows]
        if not new_cols:
            return

        # create expanded matrix
        vs_new = pd.DataFrame(0, index=old_rows + new_cols, columns=old_cols + new_cols)

        # copy old content
        for r in old_rows:
            for c in old_cols:
                vs_new.at[r, c] = old_vs.at[r, c]

        # handle newly splitted
        for c_new in new_cols:
            if c_new.endswith("_cat"):
                c_base = c_new[:-4]  # ex) 'age_cat' -> 'age'
                if c_base in vs_new.index and c_base in vs_new.columns:
                    # copy entire row from base
                    for c2 in old_cols:
                        vs_new.at[c_new, c2] = vs_new.at[c_base, c2]
                    # also copy columns in old_rows
                    for r2 in old_rows:
                        vs_new.at[r2, c_new] = vs_new.at[r2, c_base]

                vs_new.at[c_new, c_new] = 0  # no self-predict
            else:
                # brand new col => remain 0
                pass

        self.variable_selection_ = vs_new

    # -----------------------------------------------------
    # HELPER sub-routines
    # -----------------------------------------------------
    def _detect_column_order(self, X: pd.DataFrame):
        if self.syn_order:
            self.column_order_ = [c for c in self.syn_order if c in X.columns]
        else:
            self.column_order_ = list(X.columns)

    def _detect_col_types(self, X: pd.DataFrame):
        self.numeric_info_.clear()
        self.categorical_info_.clear()
        for col in X.columns:
            nuniq = X[col].nunique()
            if nuniq > self.max_categories:
                self.numeric_info_[col] = {"dtype": X[col].dtype}
            else:
                self.categorical_info_[col] = {"dtype": X[col].dtype}

    def _detect_special_values(self, X: pd.DataFrame):
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            high_vals = freq[freq > 0.9].index.tolist()
            if high_vals:
                cur = self.columns_special_values.get(col, [])
                self.columns_special_values[col] = list(cur) + high_vals

    def _reorder_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.column_order_:
            new_cols = [c for c in self.column_order_ if c in X.columns]
            return X[new_cols]
        return X

    def _split_numeric_cols(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in list(self.numeric_info_.keys()):
            if col not in X.columns:
                continue
            cat_col = col + "_cat"
            special_vals = self.columns_special_values.get(col, [])
            X[cat_col] = X[col].apply(
                lambda x: x if (x in special_vals or pd.isna(x)) else -777
            )
            X[cat_col] = X[cat_col].fillna(-9999)
            X[col] = X[col].apply(
                lambda x: x if (x not in special_vals and not pd.isna(x)) else pd.NA
            )
            X[cat_col] = X[cat_col].astype("category")
        return X

    def _update_column_types(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in X.columns:
            if col in self.numeric_info_:
                X[col] = X[col].astype(self.numeric_info_[col]["dtype"])
            elif col in self.categorical_info_:
                X[col] = X[col].astype(self.categorical_info_[col]["dtype"])
        return X

    def _assign_methods(self, X: pd.DataFrame) -> pd.DataFrame:
        self.method_assignments.clear()
        first = True
        for c in X.columns:
            if first:
                self.method_assignments[c] = "random_sampling"
                first = False
            else:
                self.method_assignments[c] = "CART"
        return X

    # -----------------------------------------------------
    # user update variable selection
    # -----------------------------------------------------
    @staticmethod
    def update_variable_selection(
        var_sel_df: pd.DataFrame,
        user_dict: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Update the variable_selection DataFrame with user_dict info:
          user_dict = {
            "D": ["B","C"],
            "A": ["B"]
          }
          => row="D", col="B","C" => 1 (others=0)
             row="A", col="B"    => 1 (others=0)

        Returns the updated var_sel_df
        """
        for target_col, predictor_list in user_dict.items():
            if target_col not in var_sel_df.index:
                print(f"[WARNING] '{target_col}' not in var_sel_df.index => skipping.")
                continue
            # set row=target_col to 0 first
            var_sel_df.loc[target_col, :] = 0
            # set 1 only for the predictor_list
            for predictor in predictor_list:
                if predictor in var_sel_df.columns:
                    var_sel_df.loc[target_col, predictor] = 1
                else:
                    print(f"[WARNING] predictor '{predictor}' not in columns => skipping.")
        return var_sel_df
