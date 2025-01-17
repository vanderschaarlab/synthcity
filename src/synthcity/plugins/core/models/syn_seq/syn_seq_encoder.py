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
      - variable_selection_ is stored internally as {target_col: [predictor_cols...]} dict
      - get_info() or debug printing can show it as a matrix (DataFrame).
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
            columns_special_values: ex) {"age":[999], "bp":[-0.04]}
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
        self.col_map: Dict[str, Dict[str, Any]] = {}
        # date minimums
        self.date_mins: Dict[str, pd.Timestamp] = {}
        # variable_selection_: internally a dictionary: { target_col: [predictors...] }
        self.variable_selection_: Optional[Dict[str, List[str]]] = None

    # ----------------- fit -----------------
    def fit(self, X: pd.DataFrame, y=None) -> "Syn_SeqEncoder":
        X = X.copy()
        self._detect_syn_order(X)
        self._init_col_map(X)
        self._detect_special_values(X)
        self._build_variable_selection_dict()  # 바뀐 부분
        self._store_date_min(X)
        return self

    # ----------------- transform -----------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = self._reorder_columns(X)
        X = self._convert_date_to_offset(X)
        X = self._split_numeric_cols_in_front(X)
        X = self._apply_converted_dtype(X)
        self._update_variable_selection_after_split_dict(X)  # 바뀐 부분
        return X

    # ----------------- inverse_transform -----------------
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1) single special => restore
        for col, specs in self.columns_special_values.items():
            if col in X.columns and len(specs) == 1:
                X[col] = X[col].replace(pd.NA, specs[0])

        # 2) offset -> date
        X = self._convert_offset_to_date(X)

        # 3) restore original_dtype
        for col in X.columns:
            if col not in self.col_map:
                continue
            orig_dt = self.col_map[col].get("original_dtype")
            if orig_dt:
                try:
                    X[col] = X[col].astype(orig_dt)
                except:
                    pass
        return X

    # ----------------- get_info() -----------------
    def get_info(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing:
          - syn_order
          - method : { col : method }
          - special_value: self.columns_special_values
          - original_type: { col : original_dtype }
          - converted_type: { col : converted_type }
          - variable_selection: for debugging, we can convert the dict -> DataFrame
        """
        info_dict: Dict[str, Any] = {}
        info_dict["syn_order"] = self.syn_order

        # method
        method_map = {}
        orig_type_map = {}
        conv_type_map = {}
        for col, cinfo in self.col_map.items():
            method_map[col] = cinfo.get("method", None)
            orig_type_map[col] = cinfo.get("original_dtype", None)
            conv_type_map[col] = cinfo.get("converted_type", None)

        info_dict["method"] = method_map
        info_dict["special_value"] = self.columns_special_values
        info_dict["original_type"] = orig_type_map
        info_dict["converted_type"] = conv_type_map

        # variable_selection_: internally a dict => convert to DataFrame for display
        if self.variable_selection_ is not None:
            df_vs = self._varsel_dict_to_matrix(self.variable_selection_)
            info_dict["variable_selection"] = df_vs
        else:
            info_dict["variable_selection"] = None

        return info_dict

    # ----------------- dictionary-based variable_selection_ -----------------
    def _build_variable_selection_dict(self) -> None:
        """
        Instead of building a DataFrame of zeros, we store a dictionary:
        For each col in syn_order => let's default to "predictors = previous columns".
        e.g. self.variable_selection_ = { "bmi": ["sex"], "bp": ["sex","bmi"], ... }
        """
        vs_dict: Dict[str, List[str]] = {}
        for i, col in enumerate(self.syn_order):
            # default: everything up to i-1 as predictor
            vs_dict[col] = self.syn_order[:i]
        self.variable_selection_ = vs_dict

    def _update_variable_selection_after_split_dict(self, X: pd.DataFrame) -> None:
        """
        If new splitted columns appear (col_cat), we decide how to handle them in dictionary-based variable_selection_.
        In the old code, we appended row/column to a DataFrame of zeros.
        Now we do it for dictionary. e.g. if "bp_cat" was created, do we treat it as new target col? Or skip?
        For demonstration, let's skip it or keep it empty. You can customize the logic as needed.
        """
        if self.variable_selection_ is None:
            return
        # check new columns not in syn_order before
        for col in X.columns:
            if col not in self.variable_selection_ and col in self.syn_order:
                # newly splitted col => we can decide default
                idx = self.syn_order.index(col)
                self.variable_selection_[col] = self.syn_order[:idx]

    def _varsel_dict_to_matrix(self, vs_dict: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Convert { target_col: [list_of_predictors], ... } -> DataFrame of 0/1
        row = target, col = predictor
        """
        all_cols = sorted(list(vs_dict.keys()))
        # 혹은 self.syn_order를 기반으로 정렬해도 됨
        df_vs = pd.DataFrame(0, index=all_cols, columns=all_cols, dtype=int)

        for tgt_col, preds in vs_dict.items():
            for p in preds:
                if p in df_vs.columns:
                    df_vs.at[tgt_col, p] = 1
        return df_vs

    # =================================================================
    # internals for col_map, date conversions, etc. (unchanged except removing DataFrame references)
    # =================================================================
    def _detect_syn_order(self, X: pd.DataFrame):
        if not self.syn_order:
            self.syn_order = list(X.columns)
        else:
            self.syn_order = [c for c in self.syn_order if c in X.columns]

    def _init_col_map(self, X: pd.DataFrame):
        self.col_map.clear()
        for col in self.syn_order:
            if col not in X.columns:
                continue
            orig_dt = str(X[col].dtype)

            declared = self.col_type.get(col, "").lower()
            if declared == "date":
                conv_type = "numeric"
            elif declared == "numeric":
                conv_type = "numeric"
            elif declared == "category":
                conv_type = "category"
            else:
                # fallback
                if pd.api.types.is_datetime64_any_dtype(X[col]):
                    conv_type = "numeric"
                    self.col_type[col] = "date"
                else:
                    nuniq = X[col].nunique()
                    if nuniq > self.max_categories:
                        conv_type = "numeric"
                    else:
                        conv_type = "category"

            self.col_map[col] = {
                "original_dtype": orig_dt,
                "converted_type": conv_type,
                "method": self.default_method
            }

    def _detect_special_values(self, X: pd.DataFrame):
        for col in self.syn_order:
            if col not in X.columns:
                continue
            cinfo = self.col_map[col]
            if cinfo.get("converted_type") != "numeric":
                continue
            freq = X[col].value_counts(dropna=False, normalize=True)
            big_ones = freq[freq > 0.9].index.tolist()
            if big_ones:
                exist = self.columns_special_values.get(col, [])
                merged = set(exist).union(big_ones)
                self.columns_special_values[col] = list(merged)

    def _store_date_min(self, X: pd.DataFrame):
        for col in self.syn_order:
            if self.col_type.get(col) == "date":
                dt_series = pd.to_datetime(X[col], errors="coerce")
                self.date_mins[col] = dt_series.min()

    def _reorder_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in self.syn_order if c in X.columns]
        return X[keep]

    def _convert_date_to_offset(self, X: pd.DataFrame) -> pd.DataFrame:
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
        for col in list(self.syn_order):
            if col not in self.col_map:
                continue
            if self.col_map[col]["converted_type"] != "numeric":
                continue
            if col not in X.columns:
                continue

            specials = self.columns_special_values.get(col, [])
            if not specials:
                continue

            cat_col = col + "_cat"
            X[cat_col] = X[col].apply(
                lambda v: v if (pd.isna(v) or v in specials) else -777
            )
            X[cat_col] = X[cat_col].fillna(-9999)
            X[col] = X[col].apply(
                lambda v: v if (not pd.isna(v) and v not in specials) else pd.NA
            )
            X[cat_col] = X[cat_col].astype("category")

            if cat_col not in self.syn_order:
                idx = self.syn_order.index(col)
                self.syn_order.insert(idx, cat_col)

            self.col_map[cat_col] = {
                "original_dtype": None,
                "converted_type": "category",
                "method": self.default_method
            }
        return X

    def _apply_converted_dtype(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.syn_order:
            if col not in X.columns:
                continue
            cinfo = self.col_map[col]
            ctype = cinfo.get("converted_type", "category")
            if ctype == "numeric":
                X[col] = pd.to_numeric(X[col], errors="coerce")
            else:
                X[col] = X[col].astype("category")
        return X

    def _convert_offset_to_date(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.syn_order:
            if self.col_type.get(col) == "date" and col in X.columns:
                offset = pd.to_numeric(X[col], errors="coerce")
                mindt = self.date_mins.get(col, None)
                if mindt is not None:
                    X[col] = pd.to_timedelta(offset, unit="D") + mindt
        return X

    def _update_variable_selection_after_split(self, X: pd.DataFrame):
        """Deprecated -> replaced by dictionary version '_update_variable_selection_after_split_dict'."""
        pass
