# synthcity/plugins/syn_seq/syn_seq_encoder.py

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
            user_variable_selection: if not None, a user-provided variable_selection matrix.
                                     Must be a DataFrame with shape (n_cols, n_cols).
                                     Columns & index must match self.syn_order or final columns.
        """
        self.columns_special_values = columns_special_values or {}
        self.syn_order = syn_order or []
        self.max_categories = max_categories

        # for custom variable_selection
        self.user_variable_selection = user_variable_selection

        self.categorical_info_ = {}
        self.numeric_info_ = {}
        self.column_order_ = []
        self.method_assignments = {}
        self.variable_selection_ = None  # This is our main "prediction matrix"
        self.prediction_matrix = None  # optional if you want a separate copy

    def fit(self, X: pd.DataFrame, y=None) -> "Syn_SeqEncoder":
        X = X.copy()

        # 1) column order
        self._detect_column_order(X)

        # 2) col types
        self._detect_col_types(X)

        # 3) special values
        self._detect_special_values(X)

        # 4) Build an empty variable_selection_ matrix (or from user input if any)
        self._build_variable_selection_matrix(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # reorder -> numeric split -> update dtypes -> methods -> ...
        X = self._reorder_columns(X)
        X = self._split_numeric_cols(X)
        X = self._update_column_types(X)
        X = self._assign_methods(X)

        # build a new 'prediction_matrix' if needed
        self._build_or_update_prediction_matrix(X)

        # update the variable_selection_ after splitting numeric -> numeric + numeric_cat
        self._update_variable_selection_after_split(X)

        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # restore special values
        for col, vals in self.columns_special_values.items():
            if col in X.columns:
                X[col] = X[col].replace(pd.NA, vals)

        # restore dtypes
        for col, info in self.categorical_info_.items():
            if col in X.columns:
                X[col] = X[col].astype(info["dtype"])
        for col, info in self.numeric_info_.items():
            if col in X.columns:
                X[col] = X[col].astype(info["dtype"])
        return X

    # --------------------------------------------------------------------
    # BUILD variable_selection_ (prediction matrix)
    # --------------------------------------------------------------------
    def _build_variable_selection_matrix(self, X: pd.DataFrame):
        """
        If user_variable_selection is given, use that.
        Otherwise, build a default matrix e.g. for each col_i,
        set predictor=1 for col_j < col_i (previous columns in order).
        """
        n = len(self.column_order_)
        df_cols = self.column_order_

        if self.user_variable_selection is not None:
            # Use the user-provided matrix. We assume user ensured correct shape & order
            vs = self.user_variable_selection.copy()
            # But let's do a minimal check
            if vs.shape != (n, n):
                raise ValueError(f"user_variable_selection must be shape ({n},{n}). Got {vs.shape}")
            # also check index & columns match
            if list(vs.index) != df_cols or list(vs.columns) != df_cols:
                raise ValueError("user_variable_selection must have same index/columns as syn_order.")
            self.variable_selection_ = vs
        else:
            # default logic: row=target col, col=predictor col
            vs = pd.DataFrame(0, index=df_cols, columns=df_cols)
            # e.g. for target i, use all j < i
            for i in range(n):
                for j in range(i):
                    vs.iat[i, j] = 1
            self.variable_selection_ = vs

    def _build_or_update_prediction_matrix(self, X: pd.DataFrame):
        """
        If you want a separate 'prediction_matrix' from variable_selection_,
        you can build it here. For now, we just store the same structure as variable_selection_.
        But if columns changed (e.g. numeric->cat), we might adapt.
        """
        # Example: we create a new empty DataFrame with the final columns
        # but keep the same row dimension. This is a toy example, adjust as needed.
        final_cols = list(X.columns)
        n2 = len(final_cols)
        self.prediction_matrix = pd.DataFrame(0, index=final_cols, columns=final_cols)

    def _update_variable_selection_after_split(self, X: pd.DataFrame):
        """
        If numeric col 'age' was splitted => 'age','age_cat', 
        we need to update variable_selection_ to handle 'age_cat' row & col.
        In your real logic, you might want to copy the row for 'age' => 'age_cat' or set them differently.
        """
        existing_rows = set(self.variable_selection_.index)
        final_cols = list(X.columns)

        # find new splitted columns
        new_cols = [c for c in final_cols if c not in existing_rows]

        if not new_cols:
            return  # no new splitted columns => do nothing

        # We'll expand the variable_selection_ to include them
        old_vs = self.variable_selection_
        old_cols = old_vs.columns.tolist()
        old_rows = old_vs.index.tolist()

        # new size
        new_size = len(old_rows) + len(new_cols)
        vs_new = pd.DataFrame(0, index=old_rows+new_cols, columns=old_cols+new_cols)

        # copy old data
        for r in old_rows:
            for c in old_cols:
                vs_new.at[r, c] = old_vs.at[r, c]

        # any new row/col logic => e.g. for "age_cat", replicate "age" row partially or user-defined
        for c_new in new_cols:
            if c_new.endswith("_cat"):
                # assume it is splitted from c_base
                c_base = c_new[:-4]
                # copy row from c_base
                if c_base in vs_new.index and c_base in vs_new.columns:
                    for c2 in old_cols:
                        vs_new.at[c_new, c2] = vs_new.at[c_base, c2]
                    # also for old rows referencing c_base as predictor
                    for r2 in old_rows:
                        vs_new.at[r2, c_new] = vs_new.at[r2, c_base]
                # but can also tweak if you want
                # e.g. set vs_new.at[c_new, c_base] = 0 if you don't want self dependency
                vs_new.at[c_new, c_new] = 0  # typically no self-predict

            else:
                # brand new col, for this example => just 0
                pass

        self.variable_selection_ = vs_new

    # --------------------------------------------------------------------
    # HELPER sub-routines
    # --------------------------------------------------------------------
    def _detect_column_order(self, X: pd.DataFrame):
        if self.syn_order:
            self.column_order_ = list(self.syn_order)
        else:
            self.column_order_ = list(X.columns)

    def _detect_col_types(self, X: pd.DataFrame):
        self.numeric_info_ = {}
        self.categorical_info_ = {}
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
                lambda x: x if x in special_vals or pd.isna(x) else -777
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

    # --------------------------------------------------------------------
    # user update variable selection
    # --------------------------------------------------------------------

    def update_variable_selection(
        var_sel_df: pd.DataFrame,
        user_dict: dict
    ) -> pd.DataFrame:
        """
        variable_selection DataFrame을 user_dict 기준으로 갱신합니다.

        Parameters
        ----------
        var_sel_df : pd.DataFrame
            2차원 0/1 매트릭스 형태의 variable_selection.
            index = target 컬럼 목록,
            columns = predictor 컬럼 목록.

        user_dict : dict
            {
                "타겟이름": ["사용할 predictor1", "사용할 predictor2", ...],
                ...
            } 형태의 입력 예:
                {
                    "D": ["B", "C"],  # row="D", col="B","C"에만 1; 나머지는 0
                    "A": ["B"]        # row="A", col="B"만 1; 나머지는 0
                }

        Returns
        -------
        pd.DataFrame
            업데이트된 variable_selection DataFrame
        """
        # user_dict를 순회하며 variable_selection 업데이트
        for target_col, predictor_list in user_dict.items():
            # 1) target_col 이 var_sel_df의 index에 있어야만 처리
            if target_col not in var_sel_df.index:
                print(f"[WARNING] '{target_col}' not in var_sel_df.index. Skipped.")
                continue

            # 2) 해당 target row를 모두 0으로 초기화
            var_sel_df.loc[target_col, :] = 0

            # 3) predictor_list에 속한 열만 1로 설정
            for predictor in predictor_list:
                if predictor in var_sel_df.columns:
                    var_sel_df.loc[target_col, predictor] = 1
                else:
                    print(f"[WARNING] predictor '{predictor}' not in columns. Skipped.")

        return var_sel_df
