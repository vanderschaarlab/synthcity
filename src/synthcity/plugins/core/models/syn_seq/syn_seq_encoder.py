from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class Syn_SeqEncoder(TransformerMixin, BaseEstimator):
    """
    - 내부적으로 "date" = day-offset numeric 으로 변환.
    - aggregator(학습) 측에는 "numeric" or "category"로만 보이길 원함.
    - 하지만 inverse_transform 때는 원래 date로 복원해야 하므로, 
      내부에서는 "date" 정보를 별도 보관.
    """

    def __init__(
        self,
        columns_special_values: Optional[Dict[str, List]] = None,
        syn_order: Optional[List[str]] = None,
        max_categories: int = 20,
        col_type: Optional[Dict[str, str]] = None,
    ) -> None:
        self.columns_special_values = columns_special_values or {}
        self.syn_order = syn_order or []
        self.max_categories = max_categories

        # 유저 선언 { "col":"numeric"/"category"/"date" }
        # => detect 결과를 합쳐 최종 col_type[col] = "numeric"/"category"/"date"
        self.col_type: Dict[str, str] = col_type.copy() if col_type else {}

        # 원본 dtype: "float64", "int64", "object", "datetime64[ns]" ...
        self.original_dtype_map: Dict[str, str] = {}

        self.column_order_: List[str] = []
        self.variable_selection_: Optional[pd.DataFrame] = None

        # 날짜 min => offset 변환용
        self.date_mins: Dict[str, pd.Timestamp] = {}

    # ------------------- FIT -------------------
    def fit(self, X: pd.DataFrame, y=None) -> "Syn_SeqEncoder":
        X = X.copy()
        self._detect_column_order(X)
        self._detect_col_types_and_store_original(X)
        self._detect_special_values(X)
        self._build_variable_selection(X)
        self._store_date_min(X)
        return self

    # ------------------- TRANSFORM -------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # 1) reorder
        X = self._reorder_columns(X)
        # 2) date => numeric offset
        X = self._convert_date_to_offset(X)
        # 3) split numeric => col_cat
        X = self._split_numeric_cols(X)
        # 4) apply role => numeric/category
        X = self._apply_role_dtype(X)
        # 5) expand variable_selection if splitted
        self._update_variable_selection_after_split(X)
        return X

    # ------------------- INVERSE_TRANSFORM -------------------
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # 1) numeric special => single val restore
        for col, specs in self.columns_special_values.items():
            if col in X.columns and len(specs) == 1:
                X[col] = X[col].replace(pd.NA, specs[0])

        # 2) numeric offset => date
        X = self._convert_offset_to_date(X)

        # 3) restore original dtype
        for col in list(X.columns):
            if col.endswith("_cat"):
                continue
            if col in self.original_dtype_map:
                odtype = self.original_dtype_map[col]
                try:
                    X[col] = X[col].astype(odtype)
                except:
                    pass
        return X

    # ------------------- aggregator에 전달: changed_dtype -------------------
    def get_changed_dtype_map(self) -> Dict[str, str]:
        """
        여기서 "date"도 학습단계에선 "numeric" 취급해주고 싶다면,
        date -> numeric 치환해서 반환하면 됨.
        """
        dtype_map = {}
        for c, role in self.col_type.items():
            if role == "date":
                dtype_map[c] = "numeric"  # aggregator는 numeric처럼 처리
            else:
                dtype_map[c] = role
        return dtype_map

    # ===================== Internals =====================
    def _detect_column_order(self, X: pd.DataFrame):
        if self.syn_order:
            self.column_order_ = [c for c in self.syn_order if c in X.columns]
        else:
            self.column_order_ = list(X.columns)

    def _detect_col_types_and_store_original(self, X: pd.DataFrame):
        for col in self.column_order_:
            if col not in X.columns:
                continue
            # 원본 dtype 저장
            self.original_dtype_map[col] = str(X[col].dtype)

            # user가 선언?
            user_decl = self.col_type.get(col, "").lower()
            if user_decl in ("numeric","category","date"):
                self.col_type[col] = user_decl
                continue

            # fallback auto
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                self.col_type[col] = "date"
            else:
                nuniq = X[col].nunique()
                # max_categories 보다 크면 numeric
                if nuniq > self.max_categories:
                    self.col_type[col] = "numeric"
                else:
                    self.col_type[col] = "category"

    def _detect_special_values(self, X: pd.DataFrame):
        for col in self.column_order_:
            if col not in X.columns:
                continue
            freq = X[col].value_counts(dropna=False, normalize=True)
            big_ones = freq[freq>0.9].index.tolist()
            if big_ones:
                exist = self.columns_special_values.get(col, [])
                merged = set(exist).union(big_ones)
                self.columns_special_values[col] = list(merged)

    def _build_variable_selection(self, X: pd.DataFrame):
        df_cols = self.column_order_
        vs = pd.DataFrame(0, index=df_cols, columns=df_cols)
        for i in range(len(df_cols)):
            for j in range(i):
                vs.iat[i,j] = 1
        self.variable_selection_ = vs

    def _store_date_min(self, X: pd.DataFrame):
        self.date_mins.clear()
        for col, role in self.col_type.items():
            if role == "date" and col in X.columns:
                dt = pd.to_datetime(X[col], errors="coerce")
                self.date_mins[col] = dt.min()

    def _reorder_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        keep_cols = [c for c in self.column_order_ if c in X.columns]
        return X[keep_cols]

    def _convert_date_to_offset(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        date -> (X[col] - mindate).dt.days
        """
        for col, role in self.col_type.items():
            if role == "date" and col in X.columns:
                X[col] = pd.to_datetime(X[col], errors="coerce")
                m = self.date_mins.get(col, None)
                if m is None:
                    m = X[col].min()
                    self.date_mins[col] = m
                X[col] = (X[col] - m).dt.days
        return X

    def _split_numeric_cols(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        numeric -> col, col_cat
        """
        for col, role in list(self.col_type.items()):
            if role != "numeric":
                continue
            if col not in X.columns:
                continue
            specials = self.columns_special_values.get(col, [])
            cat_col = col + "_cat"

            X[cat_col] = X[col].apply(
                lambda v: v if (pd.isna(v) or v in specials) else -777
            )
            X[cat_col] = X[cat_col].fillna(-9999)
            X[col] = X[col].apply(
                lambda v: v if (not pd.isna(v) and v not in specials) else pd.NA
            )
            X[cat_col] = X[cat_col].astype("category")
        return X

    def _apply_role_dtype(self, X: pd.DataFrame) -> pd.DataFrame:
        for col, role in self.col_type.items():
            if col not in X.columns:
                continue
            if role == "numeric":
                X[col] = pd.to_numeric(X[col], errors="coerce")
            elif role == "category":
                X[col] = X[col].astype("category")
            elif role == "date":
                # 이미 day-offset float
                pass
        return X

    def _convert_offset_to_date(self, X: pd.DataFrame) -> pd.DataFrame:
        for col, role in self.col_type.items():
            if role == "date" and col in X.columns:
                m = self.date_mins.get(col, None)
                if m is None:
                    continue
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X[col] = pd.to_timedelta(X[col], unit="D") + m
        return X

    def _update_variable_selection_after_split(self, X: pd.DataFrame):
        if self.variable_selection_ is None:
            return
        old_vs = self.variable_selection_
        old_rows = list(old_vs.index)
        old_cols = list(old_vs.columns)

        new_cols = [c for c in X.columns if c not in old_rows]
        if not new_cols:
            return
        vs_new = pd.DataFrame(0, index=old_rows+new_cols, columns=old_cols+new_cols)
        for r in old_rows:
            for c in old_cols:
                vs_new.at[r, c] = old_vs.at[r, c]

        for c_new in new_cols:
            if c_new.endswith("_cat"):
                c_base = c_new[:-4]
                if c_base in vs_new.index and c_base in vs_new.columns:
                    for c2 in old_cols:
                        vs_new.at[c_new, c2] = vs_new.at[c_base, c2]
                    for r2 in old_rows:
                        vs_new.at[r2, c_new] = vs_new.at[r2, c_base]
                vs_new.at[c_new, c_new] = 0

        self.variable_selection_ = vs_new

    @staticmethod
    def update_variable_selection(
        var_sel_df: pd.DataFrame,
        user_dict: Dict[str, List[str]]
    ) -> pd.DataFrame:
        for tgt, preds in user_dict.items():
            if tgt not in var_sel_df.index:
                print(f"[WARNING] {tgt} not in var_sel_df => skip")
                continue
            var_sel_df.loc[tgt, :] = 0
            for p in preds:
                if p in var_sel_df.columns:
                    var_sel_df.loc[tgt, p] = 1
                else:
                    print(f"[WARNING] predictor {p} not in columns => skip")
        return var_sel_df
