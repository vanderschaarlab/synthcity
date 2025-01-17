from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class Syn_SeqEncoder(TransformerMixin, BaseEstimator):
    """
    A minimal version of the syn_seq encoder that:
      - Maintains col_map with { original_dtype, converted_type, method } for each column
      - For date columns, does day-offset numeric transformation => converted_type="numeric"
      - Splits numeric columns into col + col_cat, placing col_cat before original col in syn_order
      - variable_selection_ is stored as {target_col: [predictor_cols...]} (dict).
      - get_info() 시 variable_selection은 dict->DataFrame 변환해 보여줄 수 있음.
      - inverse_transform() 로 date/원본 dtype 복원.
    """

    def __init__(
        self,
        columns_special_values: Optional[Dict[str, List]] = None,
        syn_order: Optional[List[str]] = None,
        max_categories: int = 20,
        col_type: Optional[Dict[str, str]] = None,
        default_method: str = "cart",
        # variable_selection을 굳이 init에서 직접 받지 않아도 됨. 필요하면 DataLoader에서 encoder.variable_selection_에 바로 할당 가능
    ) -> None:
        """
        Args:
            columns_special_values: 예: {"bp":[-0.04, -0.01]}
            syn_order: 컬럼 처리 순서
            max_categories: auto category vs numeric 구분 임계값
            col_type: { "age":"category", "birthdate":"date"... } 등 사용자 override
            default_method: col_map에 기록될 default method
        """
        self.columns_special_values = columns_special_values or {}
        self.syn_order = syn_order or []
        self.max_categories = max_categories
        self.col_type = (col_type or {}).copy()  # user override
        self.default_method = default_method

        # col_map: { col: {"original_dtype":..., "converted_type":..., "method":...}, ...}
        self.col_map: Dict[str, Dict[str, Any]] = {}

        # date min 저장용
        self.date_mins: Dict[str, pd.Timestamp] = {}

        # variable_selection_: dict. 예: {"bp":["sex","bmi"], "target":["sex","bmi","bp"], ...}
        self.variable_selection_: Optional[Dict[str, List[str]]] = None

    # ----------------------- fit -----------------------
    def fit(self, X: pd.DataFrame, y=None) -> "Syn_SeqEncoder":
        X = X.copy()
        self._detect_syn_order(X)
        self._init_col_map(X)
        self._detect_special_values(X)

        # 만약 사용자가 미리 variable_selection_을 세팅하지 않았다면, 기본 규칙으로 생성
        if self.variable_selection_ is None:
            self.variable_selection_ = self._build_varsel_dict_default()
        # date mins
        self._store_date_min(X)
        return self

    # ----------------------- transform -----------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = self._reorder_columns(X)
        X = self._convert_date_to_offset(X)
        X = self._split_numeric_cols_in_front(X)
        X = self._apply_converted_dtype(X)

        # 새 col이 생겼을 경우 variable_selection_ dict 업데이트
        self._update_varsel_dict_after_split(X)
        return X

    # ----------------------- inverse_transform -----------------------
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # (1) single special => restore
        for col, specs in self.columns_special_values.items():
            if col in X.columns and len(specs) == 1:
                X[col] = X[col].replace(pd.NA, specs[0])
        # (2) offset -> date
        X = self._convert_offset_to_date(X)
        # (3) restore original dtype
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

    # ----------------------- get_info() -----------------------
    def get_info(self) -> Dict[str, Any]:
        """
        Returns dict with:
         - syn_order
         - method (col->method)
         - special_value
         - original_type (col-> str)
         - converted_type(col-> str)
         - variable_selection => dict->DataFrame 변환
        """
        info_dict: Dict[str, Any] = {}
        info_dict["syn_order"] = self.syn_order

        # method/original_type/converted_type
        method_map = {}
        orig_type_map = {}
        conv_type_map = {}
        for c, cinfo in self.col_map.items():
            method_map[c] = cinfo.get("method")
            orig_type_map[c] = cinfo.get("original_dtype")
            conv_type_map[c] = cinfo.get("converted_type")

        info_dict["method"] = method_map
        info_dict["special_value"] = self.columns_special_values
        info_dict["original_type"] = orig_type_map
        info_dict["converted_type"] = conv_type_map
        info_dict["variable_selection"] = self.variable_selection_

        # variable_selection_: dict -> DataFrame
        # if self.variable_selection_ is not None:
        #     df_vs = self._varsel_dict_to_df(self.variable_selection_)
        #     info_dict["variable_selection"] = df_vs
        # else:
        #     info_dict["variable_selection"] = None

        return info_dict

    # ------------------- variable_selection dict helpers -------------------
    def _build_varsel_dict_default(self) -> Dict[str, List[str]]:
        """
        기본 규칙: syn_order에서 i번째 컬럼은 앞의 i개 컬럼을 predictor로.
        예: col[0] => [], col[1] => [col[0]], col[2] => [col[0], col[1]] ...
        """
        vs_dict: Dict[str, List[str]] = {}
        for i, col in enumerate(self.syn_order):
            vs_dict[col] = self.syn_order[:i]
        return vs_dict

    def _update_varsel_dict_after_split(self, X: pd.DataFrame) -> None:
        """
        split_numeric_cols_in_front(...) 로 "xx_cat" 같은 새 컬럼이 생겼을 때,
        variable_selection_ dict에도 반영.
        default로: 새 컬럼 idx = syn_order.index(새컬럼) 이전 것들을 predictor로 할당
        """
        if self.variable_selection_ is None:
            return

        for col in X.columns:
            if col not in self.variable_selection_ and col in self.syn_order:
                # 새로 생긴 컬럼. index 찾기
                idx = self.syn_order.index(col)
                # 앞의 idx개를 predictor로
                self.variable_selection_[col] = self.syn_order[:idx]

    def _varsel_dict_to_df(
        self, vs_dict: Dict[str, List[str]], syn_order: List[str]
    ) -> pd.DataFrame:
        """
        dict -> DataFrame(0/1) 변환, row/col 순서를 syn_order로 맞춘다.
        """
        # 1) syn_order 기준으로 빈 0행렬 생성
        df_vs = pd.DataFrame(0, index=syn_order, columns=syn_order, dtype=int)

        # 2) dict에서 (target_col, predictor_list) 를 읽어서 1 설정
        for tgt, preds in vs_dict.items():
            if tgt not in syn_order:
                continue  # 만약 syn_order 밖의 key면 건너뜀
            for p in preds:
                if p in syn_order:
                    df_vs.at[tgt, p] = 1
        return df_vs

    # ------------------- internal col_map, date conversions, etc. -------------------
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
                mindt = self.date_mins.get(col)
                X[col] = pd.to_datetime(X[col], errors="coerce")
                if mindt is None:
                    mindt = X[col].min()
                    self.date_mins[col] = mindt
                X[col] = (X[col] - mindt).dt.days
        return X

    def _split_numeric_cols_in_front(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in list(self.syn_order):
            info = self.col_map.get(col)
            if info is None:
                continue
            if info["converted_type"] != "numeric":
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
