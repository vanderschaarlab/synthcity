# synthcity/plugins/core/models/syn_seq/syn_seq_encoder.py

from typing import Optional, Dict, Any, List, Union
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

from synthcity.plugins.core.models.feature_encoder import DatetimeEncoder

class Syn_SeqEncoder(TransformerMixin, BaseEstimator):
    """
    Syn_SeqEncoder handles preprocessing and postprocessing tasks using fit/transform pattern,
    plus manages a 'variable_selection' matrix (like a prediction matrix).
    
    This encoder:
      - Orders columns (if syn_order specified).
      - Attempts to detect if a column is numeric, category, or date (or uses the user’s col_type).
      - Splits numeric columns into a “clean numeric” + a “_cat” column for special-value or missing marking.
      - Builds a default or user-provided variable_selection matrix (row=target, col=predictor).
      - Assigns placeholder methods (so that we have some field to see them). 
        (Now revised to match aggregator’s recognized method strings like “SWR” and “CART”.)
    """

    def __init__(
        self,
        columns_special_values: Optional[Dict[str, Any]] = None,
        syn_order: Optional[List[str]] = None,
        max_categories: int = 20,
        user_variable_selection: Optional[pd.DataFrame] = None,
        # user-declared column types (e.g. {"C1":"category","N1":"numeric","D1":"date"})
        col_type: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Args:
            columns_special_values : {colName : [specialVals...]}
            syn_order : optional column order to use
            max_categories : threshold for deciding numeric vs categorical
            user_variable_selection : optional DataFrame(n x n) with row=target, col=predictor
            col_type : user-declared column types, e.g. {"C2":"category","N1":"numeric","D1":"date"}
        """
        self.columns_special_values = columns_special_values or {}
        self.syn_order = syn_order or []
        self.max_categories = max_categories
        self.user_variable_selection = user_variable_selection

        # track info about columns
        self.categorical_info_: Dict[str, Dict[str, Any]] = {}
        self.numeric_info_: Dict[str, Dict[str, Any]] = {}
        # We'll track date columns separately
        self.date_info_: Dict[str, Dict[str, Any]] = {}

        self.column_order_: List[str] = []
        self.method_assignments: Dict[str, str] = {}
        self.variable_selection_: Optional[pd.DataFrame] = None

        # store user col_type
        self.col_type = col_type or {}

    def fit(self, X: pd.DataFrame, y=None) -> "Syn_SeqEncoder":
        # copy X to avoid side effects
        X = X.copy()

        self._detect_column_order(X)
        self._detect_col_types(X)
        self._detect_special_values(X)
        self._build_variable_selection_matrix(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1) reorder columns
        X = self._reorder_columns(X)
        # 2) split numeric => numeric + numeric_cat
        X = self._split_numeric_cols(X)
        # 3) update dtypes (including date if declared)
        X = self._update_column_types(X)
        # 4) assign placeholder methods (aligned with aggregator’s method names)
        X = self._assign_methods(X)
        # 5) update variable_selection for newly created columns
        self._update_variable_selection_after_split(X)

        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1) restore special values for numeric/date
        for col, vals in self.columns_special_values.items():
            if col in X.columns:
                # if a single special value, use it directly
                if isinstance(vals, list) and len(vals) == 1:
                    vals = vals[0]
                X[col] = X[col].replace(pd.NA, vals)

        # 2) restore dtype for categorical
        for col, info in self.categorical_info_.items():
            if col in X.columns:
                X[col] = X[col].astype(info["dtype"])

        # 3) restore date columns
        for col, info in self.date_info_.items():
            if col in X.columns:
                X[col] = pd.to_datetime(X[col], errors="coerce")

        # 4) restore numeric
        for col, info in self.numeric_info_.items():
            if col in X.columns:
                X[col] = X[col].astype(info["dtype"])

        return X

    # -----------------------------------------------------
    # variable_selection building
    # -----------------------------------------------------
    def _build_variable_selection_matrix(self, X: pd.DataFrame) -> None:
        """
        If user provided a variable_selection DataFrame, validate it. Otherwise,
        build a default (row=target, columns=all prior columns).
        """
        n = len(self.column_order_)
        df_cols = self.column_order_

        if self.user_variable_selection is not None:
            vs = self.user_variable_selection.copy()
            if vs.shape != (n, n):
                raise ValueError(
                    f"user_variable_selection must be shape ({n},{n}), got {vs.shape}"
                )
            # check row/col alignment
            if list(vs.index) != df_cols or list(vs.columns) != df_cols:
                raise ValueError(
                    "Mismatch in user_variable_selection index/columns vs syn_order."
                )
            self.variable_selection_ = vs
        else:
            # default: row=target col=predictor => row i uses columns [0..i-1]
            vs = pd.DataFrame(0, index=df_cols, columns=df_cols)
            for i in range(n):
                for j in range(i):
                    vs.iat[i, j] = 1
            self.variable_selection_ = vs

    def _update_variable_selection_after_split(self, X: pd.DataFrame) -> None:
        """
        If numeric columns were split (col => col + col_cat), we add the new
        columns into the variable_selection matrix. They replicate the row/col
        from the original but don't self-predict.
        """
        if self.variable_selection_ is None:
            return
        old_vs = self.variable_selection_
        old_rows = old_vs.index.tolist()
        old_cols = old_vs.columns.tolist()

        final_cols = list(X.columns)
        new_cols = [c for c in final_cols if c not in old_rows]
        if not new_cols:
            return

        vs_new = pd.DataFrame(0, index=old_rows + new_cols, columns=old_cols + new_cols)

        # copy old content
        for r in old_rows:
            for c in old_cols:
                vs_new.at[r, c] = old_vs.at[r, c]

        # handle new splitted columns
        for c_new in new_cols:
            if c_new.endswith("_cat"):
                c_base = c_new[:-4]  # e.g. "age_cat" => "age"
                if c_base in vs_new.index and c_base in vs_new.columns:
                    # copy entire row from base
                    for c2 in old_cols:
                        vs_new.at[c_new, c2] = vs_new.at[c_base, c2]
                    # copy entire column from base
                    for r2 in old_rows:
                        vs_new.at[r2, c_new] = vs_new.at[r2, c_base]

                vs_new.at[c_new, c_new] = 0  # new col doesn't predict itself
            else:
                # brand new col => remain all zeros
                pass

        self.variable_selection_ = vs_new

    # -----------------------------------------------------
    # HELPER sub-routines
    # -----------------------------------------------------
    def _detect_column_order(self, X: pd.DataFrame):
        """
        If user gave syn_order, we filter columns. Otherwise, use X.columns.
        """
        if self.syn_order:
            self.column_order_ = [c for c in self.syn_order if c in X.columns]
        else:
            self.column_order_ = list(X.columns)

    def _detect_col_types(self, X: pd.DataFrame):
        """
        For each column, decide if numeric/categorical/date, unless
        col_type overrides. Then store in numeric_info_, categorical_info_, date_info_.
        """
        self.numeric_info_.clear()
        self.categorical_info_.clear()
        self.date_info_.clear()

        for col in X.columns:
            declared_type = self.col_type.get(col, "").lower()
            if declared_type == "category":
                self.categorical_info_[col] = {"dtype": "category"}
            elif declared_type == "numeric":
                self.numeric_info_[col] = {"dtype": X[col].dtype}
            elif declared_type == "date":
                self.date_info_[col] = {"dtype": "datetime64[ns]"}
            else:
                # fallback auto-detect
                nuniq = X[col].nunique()
                # if #unique <= max_categories => treat as category
                if nuniq > self.max_categories:
                    # might be numeric or date
                    if pd.api.types.is_datetime64_any_dtype(X[col]):
                        self.date_info_[col] = {"dtype": "datetime64[ns]"}
                    else:
                        self.numeric_info_[col] = {"dtype": X[col].dtype}
                else:
                    self.categorical_info_[col] = {"dtype": "category"}

    def _detect_special_values(self, X: pd.DataFrame):
        """
        For each col, if a single value has >90% freq, consider it a "special" value
        and store in columns_special_values. The user can override or supply additional.
        """
        for col in X.columns:
            freq = X[col].value_counts(dropna=False, normalize=True)
            high_vals = freq[freq > 0.9].index.tolist()
            if high_vals:
                existing_vals = self.columns_special_values.get(col, [])
                combined = set(existing_vals).union(set(high_vals))
                self.columns_special_values[col] = list(combined)

    def _reorder_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.column_order_:
            new_cols = [c for c in self.column_order_ if c in X.columns]
            return X[new_cols]
        return X

    def _split_numeric_cols(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        For numeric columns, create an extra col for "special or missing" => col_cat
        Then in col, we set them to NA to ensure they are not used as numeric. 
        """
        for col in list(self.numeric_info_.keys()):
            if col not in X.columns:
                continue
            cat_col = f"{col}_cat"
            special_vals = self.columns_special_values.get(col, [])

            # Mark special or missing in cat_col
            X[cat_col] = X[col].apply(
                lambda x: x if (x in special_vals or pd.isna(x)) else -777
            )
            X[cat_col] = X[cat_col].fillna(-9999)  # placeholder for NA
            # In the numeric column, remove those special/NA
            X[col] = X[col].apply(
                lambda x: x if (x not in special_vals and not pd.isna(x)) else pd.NA
            )
            X[cat_col] = X[cat_col].astype("category")

        return X

    def _update_column_types(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Attempt to cast date columns to datetime,
        numeric to numeric dtype,
        categorical to category dtype.
        """
        # handle date columns first
        for col in self.date_info_:
            if col in X.columns:
                X[col] = pd.to_datetime(X[col], errors="coerce")

        # handle numeric / categorical
        for col in X.columns:
            if col in self.numeric_info_:
                X[col] = X[col].astype(self.numeric_info_[col]["dtype"])
            elif col in self.categorical_info_:
                X[col] = X[col].astype("category")

        return X

    def _assign_methods(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Assign placeholder methods to each column, 
        matching the aggregator's known method strings:
          - the aggregator typically uses "SWR" for the first column
          - default "CART" for subsequent columns
        """
        self.method_assignments.clear()
        first = True
        for c in X.columns:
            if first:
                self.method_assignments[c] = "SWR"
                first = False
            else:
                self.method_assignments[c] = "CART"
        return X

    @staticmethod
    def update_variable_selection(
        var_sel_df: pd.DataFrame,
        user_dict: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        In the variable_selection matrix, set row=target_col => 1 in the predictor columns.
          user_dict = { "D": ["B","C"], "A": ["B"] }
          => row "D", col "B","C" => 1, row "A", col "B" => 1
        """
        for target_col, predictor_list in user_dict.items():
            if target_col not in var_sel_df.index:
                print(f"[WARNING] '{target_col}' not in var_sel_df.index => skipping.")
                continue
            # set row=target_col to 0 first
            var_sel_df.loc[target_col, :] = 0
            # set 1 only for the user-specified predictors
            for pred in predictor_list:
                if pred in var_sel_df.columns:
                    var_sel_df.loc[target_col, pred] = 1
                else:
                    print(f"[WARNING] predictor '{pred}' not in columns => skipping.")
        return var_sel_df
