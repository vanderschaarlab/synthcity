# File: syn_seq_encoder.py
from typing import Optional, Dict, Any, List, Union
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class Syn_SeqEncoder(TransformerMixin, BaseEstimator):
    """
    Syn_SeqEncoder handles preprocessing and postprocessing tasks using fit/transform pattern,
    plus manages a 'variable_selection' matrix akin to a prediction matrix.

    - columns_special_values : {colName : [specialVals...]}
    - syn_order : column order
    - max_categories : threshold for deciding numeric vs categorical
    - user_variable_selection : optional DataFrame(n x n) (row=target, col=predictor)

    추가:
    - datetime_info_ : { colName: { "dtype": ..., "reference": min_val } }
      날짜/시간형 컬럼 감지 시, 최소값(min date)을 reference로 저장해두어 transform / inverse_transform에서 활용
    """

    def __init__(
        self,
        columns_special_values: Optional[Dict[str, Any]] = None,
        syn_order: Optional[List[str]] = None,
        max_categories: int = 20,
        user_variable_selection: Optional[pd.DataFrame] = None,
    ) -> None:
        self.columns_special_values = columns_special_values or {}
        self.syn_order = syn_order or []
        self.max_categories = max_categories

        # user-provided variable selection or None
        self.user_variable_selection = user_variable_selection

        self.categorical_info_: Dict[str, Dict[str, Any]] = {}
        self.numeric_info_: Dict[str, Dict[str, Any]] = {}
        self.datetime_info_: Dict[str, Dict[str, Any]] = {}  # key=col, val={"dtype":..., "reference":...}

        self.column_order_: List[str] = []
        self.method_assignments: Dict[str, str] = {}

        # main variable_selection (prediction matrix)
        self.variable_selection_: Optional[pd.DataFrame] = None

    # --------------------------------------------------------------------
    # Fit
    # --------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> "Syn_SeqEncoder":
        X = X.copy()

        # 1) 컬럼 순서 결정
        self._detect_column_order(X)
        # 2) 컬럼별로 date vs numeric vs categorical 파악
        self._detect_col_types(X)
        # 3) high_freq 등등으로 special values 업데이트
        self._detect_special_values(X)
        # 4) variable_selection matrix (prediction matrix) 구성
        self._build_variable_selection_matrix(X)

        return self

    # --------------------------------------------------------------------
    # Transform
    # --------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1) 날짜 -> numeric 변환 ( (X[col] - reference).days )
        X = self._datetime_to_numeric(X)
        # 2) reorder columns
        X = self._reorder_columns(X)
        # 3) split numeric -> numeric + numeric_cat
        X = self._split_numeric_cols(X)
        # 4) update dtypes
        X = self._update_column_types(X)
        # 5) assign methods
        X = self._assign_methods(X)
        # 6) update variable_selection after splitting numeric
        self._update_variable_selection_after_split(X)

        return X

    # --------------------------------------------------------------------
    # inverse_transform
    # --------------------------------------------------------------------
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1) restore special values
        for col, vals in self.columns_special_values.items():
            if col in X.columns:
                # 만약 vals가 list이고 한 개뿐이라면 해당 값만 치환
                if isinstance(vals, list) and len(vals) == 1:
                    vals = vals[0]
                X[col] = X[col].replace(pd.NA, vals)

        # 2) numeric -> date 복원
        X = self._numeric_to_datetime(X)

        # 3) restore original dtypes (categorical, numeric 등)
        for col, info in self.categorical_info_.items():
            if col in X.columns:
                X[col] = X[col].astype(info["dtype"])
        for col, info in self.numeric_info_.items():
            if col in X.columns:
                X[col] = X[col].astype(info["dtype"])

        return X

    # --------------------------------------------------------------------
    # variable_selection (prediction matrix) building
    # --------------------------------------------------------------------
    def _build_variable_selection_matrix(self, X: pd.DataFrame) -> None:
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
            # default: row=target, col=predictor
            vs = pd.DataFrame(0, index=df_cols, columns=df_cols)
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

        # expand the matrix
        vs_new = pd.DataFrame(0, index=old_rows + new_cols, columns=old_cols + new_cols)

        # copy old content
        for r in old_rows:
            for c in old_cols:
                vs_new.at[r, c] = old_vs.at[r, c]

        # handle splitted numeric => col, col_cat
        for c_new in new_cols:
            if c_new.endswith("_cat"):
                c_base = c_new[:-4]
                if c_base in vs_new.index and c_base in vs_new.columns:
                    # copy entire row from c_base
                    for c2 in old_cols:
                        vs_new.at[c_new, c2] = vs_new.at[c_base, c2]
                    # copy column in old_rows
                    for r2 in old_rows:
                        vs_new.at[r2, c_new] = vs_new.at[r2, c_base]

                # self-predict = 0
                vs_new.at[c_new, c_new] = 0

        self.variable_selection_ = vs_new

    # --------------------------------------------------------------------
    # HELPER sub-routines
    # --------------------------------------------------------------------
    def _detect_column_order(self, X: pd.DataFrame):
        if self.syn_order:
            self.column_order_ = [c for c in self.syn_order if c in X.columns]
        else:
            self.column_order_ = list(X.columns)

    def _detect_col_types(self, X: pd.DataFrame):
        self.numeric_info_.clear()
        self.categorical_info_.clear()
        self.datetime_info_.clear()

        for col in X.columns:
            col_dtype = X[col].dtype

            if col_dtype.kind == "M":
                # datetime
                min_val = pd.to_datetime(X[col].min(), errors="coerce")
                if pd.isna(min_val):
                    min_val = pd.Timestamp("1970-01-01")
                self.datetime_info_[col] = {
                    "dtype": col_dtype,
                    "reference": min_val,
                }
            else:
                # numeric vs categorical
                nuniq = X[col].nunique()
                if nuniq > self.max_categories:
                    self.numeric_info_[col] = {"dtype": col_dtype}
                else:
                    self.categorical_info_[col] = {"dtype": col_dtype}

    def _detect_special_values(self, X: pd.DataFrame):
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            high_vals = freq[freq > 0.9].index.tolist()
            if high_vals:
                cur = self.columns_special_values.get(col, [])
                self.columns_special_values[col] = list(cur) + high_vals

    def _datetime_to_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        날짜형 -> (days) float 변환. reference(최소값)를 0일차로 본다.
        """
        for col, info in self.datetime_info_.items():
            if col in X.columns:
                ref_date = info["reference"]
                # datetime -> days float
                X[col] = (pd.to_datetime(X[col]) - ref_date).dt.days.astype(float)
        return X

    def _numeric_to_datetime(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        float(일수) -> 다시 날짜(datetime) 복원
        """
        for col, info in self.datetime_info_.items():
            if col in X.columns:
                ref_date = info["reference"]
                
                def to_date(x):
                    if pd.isna(x):
                        return pd.NA
                    return ref_date + pd.Timedelta(days=float(x))

                X[col] = X[col].apply(to_date)
                X[col] = pd.to_datetime(X[col], errors="coerce")
        return X

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
                lambda val: val if (val in special_vals or pd.isna(val)) else -777
            )
            X[cat_col] = X[cat_col].fillna(-9999)
            X[col] = X[col].apply(
                lambda val: val if (val not in special_vals and not pd.isna(val)) else pd.NA
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
    # User update variable selection
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
            # set 1 only for predictor_list
            for predictor in predictor_list:
                if predictor in var_sel_df.columns:
                    var_sel_df.loc[target_col, predictor] = 1
                else:
                    print(f"[WARNING] predictor '{predictor}' not in columns => skipping.")
        return var_sel_df
