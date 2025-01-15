from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

class Syn_SeqEncoder(TransformerMixin, BaseEstimator):
    """
    A minimal version of the syn_seq encoder that:
      - Maintains col_map with { original_dtype, converted_type, method } for each column
      - For date columns, does day-offset numeric transformation => converted_type="numeric"
      - Splits numeric columns into col + col_cat, with col_cat placed before the original col in syn_order
      - Expands variable_selection_ accordingly
      - inverse_transform can restore date columns from day-offset,
        and attempt to restore original_dtype if possible
    """

    def __init__(
        self,
        columns_special_values: Optional[Dict[str, List]] = None,
        syn_order: Optional[List[str]] = None,
        max_categories: int = 20,
        col_type: Optional[Dict[str, str]] = None,
        default_method: str = "cart",
    ) -> None:
        """
        Args:
            columns_special_values: { colName : [specialVals...] } ex) {"age":[999], "bp":[-0.04]}
            syn_order: column order to follow
            max_categories: threshold for deciding numeric vs category if not declared
            col_type: user overrides => { "birthdate":"date","sex":"category","bp":"numeric"... }
            default_method: default method name for newly recognized columns
        """
        self.columns_special_values = columns_special_values or {}
        self.syn_order = syn_order or []
        self.max_categories = max_categories
        self.col_type = (col_type or {}).copy()  # user override
        self.default_method = default_method

        # col_map: each col -> { "original_dtype", "converted_type", "method" }
        # converted_type ∈ {"numeric","category"}
        self.col_map: Dict[str, Dict[str, Any]] = {}

        # date minimums for offset
        self.date_mins: Dict[str, pd.Timestamp] = {}

        # variable_selection matrix
        self.variable_selection_: Optional[pd.DataFrame] = None

    # ----------------------------------------------------------------
    # Fit
    # ----------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> "Syn_SeqEncoder":
        X = X.copy()
        # 1) Decide syn_order (if not given)
        self._detect_syn_order(X)
        # 2) init col_map
        self._init_col_map(X)   
        # 3) detect major special values
        self._detect_special_values(X)
        # 4) build default variable_selection
        self._build_variable_selection(X)
        # 5) store date mins
        self._store_date_min(X)
        return self

    # ----------------------------------------------------------------
    # Transform
    # ----------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1) reorder columns
        X = self._reorder_columns(X)
        # 2) date -> day offset => converted_type="numeric"
        X = self._convert_date_to_offset(X)
        # 3) numeric -> split col_cat, place col_cat before original col
        X = self._split_numeric_cols_in_front(X)
        # 4) apply final dtype => numeric or category
        X = self._apply_converted_dtype(X)
        # 5) expand variable_selection if splitted col
        self._update_variable_selection_after_split(X)
        return X

    # ----------------------------------------------------------------
    # inverse_transform
    # ----------------------------------------------------------------
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1) single special => restore
        for col, specs in self.columns_special_values.items():
            if col in X.columns and len(specs)==1:
                X[col] = X[col].replace(pd.NA, specs[0])

        # 2) offset -> date
        X = self._convert_offset_to_date(X)

        # 3) restore original_dtype if possible
        for col in X.columns:
            if col not in self.col_map:
                # splitted col?
                continue
            orig_dt = self.col_map[col]["original_dtype"]
            if orig_dt is not None:
                try:
                    X[col] = X[col].astype(orig_dt)
                except:
                    pass
        return X

    # ----------------------------------------------------------------
    # set_method / get_method : if aggregator or user wants to override
    # ----------------------------------------------------------------
    def set_method(self, col: str, method: str) -> None:
        """Override method for a column."""
        if col not in self.col_map:
            print(f"[WARNING] col_map has no '{col}'")
            return
        self.col_map[col]["method"] = method

    def get_method_map(self) -> Dict[str, str]:
        """Return { col : method } for all columns in col_map."""
        return { c: info["method"] for c, info in self.col_map.items() }

    # =================================================================
    # Internals
    # =================================================================
    def _detect_syn_order(self, X: pd.DataFrame):
        if not self.syn_order:
            self.syn_order = list(X.columns)
        else:
            self.syn_order = [c for c in self.syn_order if c in X.columns]

    def _init_col_map(self, X: pd.DataFrame):
        """
        For each col in syn_order, decide if numeric or category or date => eventually store converted_type
         - date => will become numeric in transform, but we'll keep track for inverse
        """
        self.col_map.clear()

        for col in self.syn_order:
            if col not in X.columns:
                continue
            orig_dt = str(X[col].dtype)

            # Decide if date => (converted_type="numeric"), else numeric/category
            declared = self.col_type.get(col, "").lower()
            if declared == "date":
                # We'll treat as numeric, but keep date info for inverse
                conv_type = "numeric"
            elif declared == "numeric":
                conv_type = "numeric"
            elif declared == "category":
                conv_type = "category"
            else:
                # fallback auto
                if pd.api.types.is_datetime64_any_dtype(X[col]):
                    conv_type = "numeric"  # internally day-offset
                    self.col_type[col] = "date"  # mark for inverse
                else:
                    nuniq = X[col].nunique()
                    if nuniq > self.max_categories:
                        conv_type = "numeric"
                    else:
                        conv_type = "category"

            # save to col_map
            self.col_map[col] = {
                "original_dtype": orig_dt,
                "converted_type": conv_type,
                "method": self.default_method
            }

    def _detect_special_values(self, X: pd.DataFrame):
        for col in self.syn_order:
            if col not in X.columns:
                continue
            freq = X[col].value_counts(dropna=False, normalize=True)
            big_ones = freq[freq>0.9].index.tolist()
            if big_ones:
                exist = self.columns_special_values.get(col, [])
                merged = set(exist).union(big_ones)
                self.columns_special_values[col] = list(merged)

    def _build_variable_selection(self, X: pd.DataFrame):
        idx = self.syn_order
        vs = pd.DataFrame(0, index=idx, columns=idx)
        for i in range(len(idx)):
            for j in range(i):
                vs.iat[i,j] = 1
        self.variable_selection_ = vs

    def _store_date_min(self, X: pd.DataFrame):
        """For date columns => store min date for offset calc"""
        for col in self.syn_order:
            # if user said date or we auto-detected => self.col_type[col]=="date"
            # but in col_map => "converted_type":"numeric"
            # we can check self.col_type if "date"
            if self.col_type.get(col) == "date":
                dt_series = pd.to_datetime(X[col], errors="coerce")
                self.date_mins[col] = dt_series.min()

    def _reorder_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in self.syn_order if c in X.columns]
        return X[keep]

    def _convert_date_to_offset(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        If col_type[col]=="date", then X[col] => day offset vs min
        col_map[col]["converted_type"] stays "numeric"
        """
        for col in self.syn_order:
            if self.col_type.get(col) == "date" and col in X.columns:
                mindt = self.date_mins.get(col, None)
                X[col] = pd.to_datetime(X[col], errors="coerce")
                if mindt is None:
                    mindt = X[col].min()
                    self.date_mins[col] = mindt
                X[col] = (X[col] - mindt).dt.days
        return X

    def _split_numeric_cols_in_front(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        For each col whose converted_type is "numeric", create col_cat.
        Insert col_cat before col in syn_order, add to col_map
        """
        for col in list(self.syn_order):
            if col not in self.col_map:
                continue
            if self.col_map[col]["converted_type"] != "numeric":
                continue
            if col not in X.columns:
                continue

            specials = self.columns_special_values.get(col, [])
            if not specials:
                # specials가 비어 있으면 => skip => split하지 않음
                continue

            cat_col = col + "_cat"
            specials = self.columns_special_values.get(col, [])

            X[cat_col] = X[col].apply(
                lambda v: v if (pd.isna(v) or v in specials) else -777
            )
            X[cat_col] = X[cat_col].fillna(-9999)
            X[col] = X[col].apply(
                lambda v: v if (not pd.isna(v) and v not in specials) else pd.NA
            )
            X[cat_col] = X[cat_col].astype("category")

            # insert cat_col in syn_order
            if cat_col not in self.syn_order:
                idx = self.syn_order.index(col)
                self.syn_order.insert(idx, cat_col)

            # add to col_map
            self.col_map[cat_col] = {
                "original_dtype": None,
                "converted_type": "category",
                "method": self.default_method
            }

        return X

    def _apply_converted_dtype(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        If converted_type=="numeric" => to_numeric
        else "category" => astype("category")
        """
        for col in self.syn_order:
            if col not in X.columns:
                continue
            cinfo = self.col_map.get(col, {})
            ctype = cinfo.get("converted_type","category")
            if ctype=="numeric":
                X[col] = pd.to_numeric(X[col], errors="coerce")
            else:
                X[col] = X[col].astype("category")
        return X

    def _convert_offset_to_date(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        If self.col_type[col]=="date", => offset => date
        """
        for col in self.syn_order:
            if self.col_type.get(col)=="date" and col in X.columns:
                offset = pd.to_numeric(X[col], errors="coerce")
                mindt = self.date_mins.get(col, None)
                if mindt is not None:
                    X[col] = pd.to_timedelta(offset, unit="D") + mindt
        return X

    def _update_variable_selection_after_split(self, X: pd.DataFrame):
        if self.variable_selection_ is None:
            return
        old_vs = self.variable_selection_
        old_idx = list(old_vs.index)
        old_cols = list(old_vs.columns)
        new_cols = [c for c in X.columns if c not in old_idx]
        if not new_cols:
            return

        vs_new = pd.DataFrame(0, index=old_idx+new_cols, columns=old_cols+new_cols)
        for r in old_idx:
            for c in old_cols:
                vs_new.at[r,c] = old_vs.at[r,c]

        for c_new in new_cols:
            if c_new.endswith("_cat"):
                c_base = c_new[:-4]
                # copy row/col from base
                if c_base in vs_new.index and c_base in vs_new.columns:
                    for c2 in old_cols:
                        vs_new.at[c_new, c2] = vs_new.at[c_base, c2]
                    for r2 in old_idx:
                        vs_new.at[r2, c_new] = vs_new.at[r2, c_base]
                vs_new.at[c_new, c_new] = 0

        self.variable_selection_ = vs_new

    # ------------------ update_variable_selection
    @staticmethod
    def update_variable_selection(
        var_sel_df: pd.DataFrame,
        user_dict: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        For row=target, col= predictor => set 1
        """
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
