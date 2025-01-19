# File: syn_seq_encoder.py

from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

class Syn_SeqEncoder(TransformerMixin, BaseEstimator):
    """
    Handles column-by-column encoding logic for syn_seq plugin:
      - Maintains col_map with { original_dtype, converted_type, method } for each column
      - Splits numeric columns with user-defined or auto-detected special values (aka "categorization").
      - Maintains user-specified or auto variable_selection_ logic (predictor sets).
      - Maintains assigned method per column, defaulting first col => "swr", others => "cart" unless user overrides.
      - For date columns, does day-offset numeric transform => converted_type="numeric".
      - get_info() can display final col_map, variable_selection, date_mins, etc.
    """

    def __init__(
        self,
        columns_special_values: Optional[Dict[str, List]] = None,
        syn_order: Optional[List[str]] = None,
        method: Optional[Dict[str, str]] = None,
        max_categories: int = 20,
        col_type: Optional[Dict[str, str]] = None,
        variable_selection: Optional[Dict[str, List]] = None,
        default_method: str = "cart",
    ) -> None:
        """
        Args:
            columns_special_values: e.g. {"bp":[-0.04, -0.01]}
            syn_order: columns order for sequential modeling
            method: user-specified method overrides, e.g. {"bp": "norm"}
            max_categories: threshold for deciding category vs numeric if not user-specified
            col_type: forced type from user, e.g. { "age":"category", "some_date":"date" }
            variable_selection: dict => each column => list of col names used as predictor
            default_method: default method for columns (besides the first one => "swr")
        """
        self.syn_order = syn_order or []
        self.method = method or {}
        self.max_categories = max_categories
        self.columns_special_values = columns_special_values or {}
        self.col_type = col_type or {}
        self.variable_selection_ = variable_selection or {}
        self.default_method = default_method

        self.col_map: Dict[str, Dict[str, Any]] = {}
        self.date_mins: Dict[str, pd.Timestamp] = {}
        self.imbalanced_or_special_cols: List[str] = []  # track columns w/ special vals
        self._is_fit = False

    def fit(self, X: pd.DataFrame) -> "Syn_SeqEncoder":
        X = X.copy()
        # Step 1) ensure syn_order covers only existing columns
        self._detect_syn_order(X)
        # Step 2) build col_map with preliminary type detection
        self._init_col_map(X)
        # Step 3) detect + unify special values, track if col is "imbalanced"
        self._detect_special_values_and_imbalance(X)
        # Step 4) store date minimum for offset logic
        self._store_date_min(X)
        # Step 5) detect or fallback to method for each column
        self._assign_method_to_cols()
        # Step 6) build or fallback variable selection
        self._assign_variable_selection()
        self._is_fit = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fit:
            raise RuntimeError("Encoder not fit yet. Call .fit(...) first.")
        X = X.copy()
        X = self._reorder_columns(X)
        X = self._convert_date_to_offset(X)
        # Split numeric columns (with special values) into col + col_cat
        X = self._split_numeric_cols_in_front(X)
        # Convert final dtype
        X = self._apply_converted_dtype(X)
        # Possibly update variable_selection_ to reflect new cat columns
        self._update_varsel_dict_after_split(X)
        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # (1) offset -> date
        X = self._convert_offset_to_date(X)
        # (2) restore original dtype
        for col in X.columns:
            if col in self.col_map:
                orig_dt = self.col_map[col].get("original_dtype")
                if orig_dt:
                    try:
                        X[col] = X[col].astype(orig_dt)
                    except:
                        pass
        return X

    def _detect_syn_order(self, X: pd.DataFrame) -> None:
        if not self.syn_order:
            self.syn_order = list(X.columns)
        else:
            # Keep only columns that exist in X
            self.syn_order = [c for c in self.syn_order if c in X.columns]

    def _init_col_map(self, X: pd.DataFrame) -> None:
        self.col_map.clear()
        for col in self.syn_order:
            if col not in X.columns:
                continue
            orig_dt = str(X[col].dtype)
            # user override
            declared = self.col_type.get(col, "").lower()
            if declared == "date":
                conv_type = "numeric"
            elif declared == "numeric":
                conv_type = "numeric"
            elif declared == "category":
                conv_type = "category"
            else:
                # fallback detection
                if pd.api.types.is_datetime64_any_dtype(X[col]):
                    conv_type = "numeric"
                    self.col_type[col] = "date"
                else:
                    nuniq = X[col].nunique(dropna=False)
                    if nuniq <= self.max_categories:
                        conv_type = "category"
                    else:
                        conv_type = "numeric"
            self.col_map[col] = {
                "original_dtype": orig_dt,
                "converted_type": conv_type,
                "method": None,  # assigned later
            }

    def _detect_special_values_and_imbalance(self, X: pd.DataFrame) -> None:
        # unify user-specified with auto-detected big-ones
        for col in self.syn_order:
            if col not in X.columns:
                continue
            info = self.col_map[col]
            if info["converted_type"] != "numeric":
                continue
            # user special values
            user_vals = self.columns_special_values.get(col, [])
            # auto detection if e.g. single value above 0.9 freq
            freq = X[col].value_counts(dropna=False, normalize=True)
            big_ones = freq[freq > 0.9].index.tolist()  # threshold
            final_sv = sorted(set(user_vals).union(set(big_ones)))
            if final_sv:
                self.columns_special_values[col] = final_sv
                self.imbalanced_or_special_cols.append(col)

    def _store_date_min(self, X: pd.DataFrame) -> None:
        for col in self.syn_order:
            if self.col_type.get(col) == "date":
                dt_series = pd.to_datetime(X[col], errors="coerce")
                self.date_mins[col] = dt_series.min()

    def _assign_method_to_cols(self) -> None:
        """
        For each col in syn_order:
          - if index == 0 => method is "swr" unless user override
          - else => user method override if any, else default_method
        """
        for i, col in enumerate(self.syn_order):
            chosen_method = None
            if col in self.method:  # user override
                chosen_method = self.method[col]
            else:
                if i == 0:
                    # first col => fallback "swr" if not specified
                    chosen_method = "swr"
                else:
                    chosen_method = self.default_method
            if col in self.col_map:
                self.col_map[col]["method"] = chosen_method

    def _assign_variable_selection(self) -> None:
        # fill for columns missing in user variable_selection => previous columns
        for i, col in enumerate(self.syn_order):
            if col not in self.variable_selection_:
                self.variable_selection_[col] = self.syn_order[:i]

    def _reorder_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in self.syn_order if c in X.columns]
        return X[keep]

    def _convert_date_to_offset(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.syn_order:
            if self.col_type.get(col) == "date" and col in X.columns:
                mindt = self.date_mins.get(col)
                X[col] = pd.to_datetime(X[col], errors="coerce")
                if mindt is None:
                    mindt = X[col].min()
                    self.date_mins[col] = mindt
                X[col] = (X[col] - mindt).dt.days
        return X

    def _split_numeric_cols_in_front(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        For each numeric col with special_values, create a col_cat with those special or NaNs,
        then set original col to NaN in those places.
        Insert col_cat right before the original col in syn_order.
        """
        original_order = list(self.syn_order)  # copy
        for col in original_order:
            if col not in X.columns:
                continue
            info = self.col_map.get(col, {})
            if info.get("converted_type") != "numeric":
                continue
            specials = self.columns_special_values.get(col, [])
            if not specials:
                continue

            cat_col = col + "_cat"
            # build cat col with special or None => label them with e.g. { -9999 => NA special, or the actual special val code? }
            def cat_mapper(v):
                if pd.isna(v):
                    return -9999
                if v in specials:
                    return v
                return -777  # for normal values

            X[cat_col] = X[col].apply(cat_mapper)
            X[cat_col] = X[cat_col].astype("category")

            # now set original col to NaN where it's special
            def numeric_mapper(v):
                if pd.isna(v):
                    return np.nan
                if v in specials:
                    return np.nan
                return v

            X[col] = X[col].apply(numeric_mapper)

            # insert cat_col into syn_order
            if cat_col not in self.syn_order:
                idx = self.syn_order.index(col)
                self.syn_order.insert(idx, cat_col)

            # col_map entry for cat_col
            self.col_map[cat_col] = {
                "original_dtype": None,
                "converted_type": "category",
                "method": None,  # assigned later
            }
            # Now fix the method for cat_col => the same as the base col or "cart"?
            # By default, let's do same as base col, or fallback "cart"
            base_method = self.col_map[col].get("method", "cart")
            self.col_map[cat_col]["method"] = base_method

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
        # revert numeric offsets to date if col_type = date
        for col in self.syn_order:
            if self.col_type.get(col) == "date" and col in X.columns:
                offset = pd.to_numeric(X[col], errors="coerce")
                mindt = self.date_mins.get(col, None)
                if mindt is not None:
                    X[col] = pd.to_timedelta(offset, unit="D") + mindt
        return X

    def _update_varsel_dict_after_split(self, X: pd.DataFrame) -> None:
        # If splitting created col_cat => we want them also in varsel if base col was
        new_splits = {}
        for col in self.syn_order:
            if col.endswith("_cat"):
                base = col[:-4]
                if base in self.col_map:
                    new_splits[base] = col

        # for columns newly formed that aren't in variable_selection_, fill default
        for col in X.columns:
            if col not in self.variable_selection_ and col in self.syn_order:
                idx = self.syn_order.index(col)
                self.variable_selection_[col] = self.syn_order[:idx]

        # if base col was in varsel => cat col also included
        for tgt_col, pred_list in self.variable_selection_.items():
            updated = set(pred_list)
            for base_col, cat_col in new_splits.items():
                if base_col in updated:
                    updated.add(cat_col)
            self.variable_selection_[tgt_col] = list(updated)

    def get_info(self) -> Dict[str, Any]:
        """
        Returns a dictionary summarizing the encoder config.
        For printing, you can parse the dictionary or transform to a nice layout.
        """
        return {
            "syn_order": self.syn_order,
            "col_map": self.col_map,
            "variable_selection": self.variable_selection_,
            "columns_special_values": self.columns_special_values,
            "date_mins": self.date_mins,
            "imbalanced_or_special_cols": list(set(self.imbalanced_or_special_cols)),
        }
