from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

class Syn_SeqEncoder(TransformerMixin, BaseEstimator):
    """
    syn_seq용 column-by-column 인코더.
      - fit: 전처리 준비 (col_map, variable_selection 설정 등)
      - transform: 실제 변환 (date->offset, numeric->split, dtype 적용, etc.)
      - get_info: 고정된 키 이름으로 딕트 반환
        => "syn_order", "original_dtype", "converted_type", "method", 
           "special_value"(dict형), "date_mins", "variable_selection"
    """

    def __init__(
        self,
        special_value: Optional[Dict[str, List]] = None,  # 기존 columns_special_values => special_value 로 네이밍 변경
        syn_order: Optional[List[str]] = None,
        method: Optional[Dict[str, str]] = None,
        max_categories: int = 20,
        col_type: Optional[Dict[str, str]] = None,
        variable_selection: Optional[Dict[str, List]] = None,
        default_method: str = "cart",
    ):
        """
        Args:
            special_value: 예) {"bp":[-0.04, -0.01]} + auto로 감지된 freq>0.9 값도 합쳐 저장
            syn_order: 순차적 모델링 순서
            method: 유저 오버라이드 {"bp":"norm", ...}
            max_categories: 범주 vs 수치 구분 임계값
            col_type: {"age":"category","some_date":"date"} 등 명시
            variable_selection: { col: [predictors...] }
            default_method: 첫 컬럼 제외 기본값 "cart" (첫 컬럼은 없으면 "swr")
        """
        self.syn_order = syn_order or []
        self.method = method or {}
        self.max_categories = max_categories
        self.col_type = col_type or {}
        self.variable_selection_ = variable_selection or {}
        self.default_method = default_method

        # special_value: col -> list of special vals
        self.special_value: Dict[str, List] = special_value or {}

        # col_map: {col: {"original_dtype":"float64","converted_type":"numeric","method":"cart"}, ...}
        self.col_map: Dict[str, Dict[str, Any]] = {}
        self.date_mins: Dict[str, pd.Timestamp] = {}

        self._is_fit = False

    def fit(self, X: pd.DataFrame) -> "Syn_SeqEncoder":
        X = X.copy()
        # 1) syn_order 정리
        self._detect_syn_order(X)
        # 2) col_map 초기화
        self._init_col_map(X)
        # 3) special value 감지(유저+freq>0.9) → self.special_value에 통합
        self._detect_special_values(X)
        # 4) date min 기록
        self._store_date_min(X)
        # 5) method 할당
        self._assign_method_to_cols()
        # 6) variable_selection 세팅(유저 없으면 기본)
        self._assign_variable_selection()

        self._is_fit = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fit:
            raise RuntimeError("Must fit() first.")

        X = X.copy()
        X = self._reorder_columns(X)
        X = self._convert_date_to_offset(X)
        X = self._split_numeric_cols_in_front(X)  # special_value에 맞춰 _cat 생성
        X = self._apply_converted_dtype(X)
        self._update_varsel_dict_after_split(X)
        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # date offset → date 복원
        X = self._convert_offset_to_date(X)
        # original dtype 복원
        for col in X.columns:
            if col in self.col_map:
                orig = self.col_map[col].get("original_dtype")
                if orig:
                    try:
                        X[col] = X[col].astype(orig)
                    except:
                        pass
        return X

    # ---------------------- (fit) helpers ----------------------
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
                # auto
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
                "method": None
            }

    def _detect_special_values(self, X: pd.DataFrame):
        """
        fit 시점에서 user + auto(freq>0.9) special values 합치기
        => self.special_value[col] = [... all special vals ...]
        """
        for col in self.syn_order:
            if col not in X.columns:
                continue
            info = self.col_map[col]
            if info["converted_type"] != "numeric":
                continue

            # 기존에 user가 준 값
            user_vals = self.special_value.get(col, [])

            # auto: freq>0.9
            freq = X[col].value_counts(dropna=False, normalize=True)
            big_ones = freq[freq > 0.9].index.tolist()

            merged = sorted(set(user_vals).union(set(big_ones)))
            if merged:
                self.special_value[col] = merged
            else:
                # 만약 아무도 없으면 굳이 빈 list로 남김
                if col in self.special_value:
                    # user가 줬는데 empty? => 그대로 둘 수도
                    pass

    def _store_date_min(self, X: pd.DataFrame):
        for col in self.syn_order:
            if self.col_type.get(col) == "date":
                arr = pd.to_datetime(X[col], errors="coerce")
                self.date_mins[col] = arr.min()

    def _assign_method_to_cols(self):
        for i, col in enumerate(self.syn_order):
            user_m = self.method.get(col)
            if i == 0:
                chosen = user_m if user_m else "swr"
            else:
                chosen = user_m if user_m else self.default_method
            self.col_map[col]["method"] = chosen

    def _assign_variable_selection(self):
        for i, col in enumerate(self.syn_order):
            if col not in self.variable_selection_:
                self.variable_selection_[col] = self.syn_order[:i]

    # ---------------------- (transform) helpers ----------------------
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
        special_value[col] 에 값이 있으면 => col_cat 생성
        """
        original_order = list(self.syn_order)
        for col in original_order:
            if col not in X.columns:
                continue
            info = self.col_map.get(col, {})
            if info["converted_type"] != "numeric":
                continue
            specials = self.special_value.get(col, [])
            if not specials:
                continue

            cat_col = col + "_cat"

            def cat_mapper(v):
                if pd.isna(v):
                    return -9999
                if v in specials:
                    return v
                return -777

            X[cat_col] = X[col].apply(cat_mapper).astype("category")

            def numeric_mapper(v):
                if pd.isna(v):
                    return np.nan
                if v in specials:
                    return np.nan
                return v

            X[col] = X[col].apply(numeric_mapper)

            if cat_col not in self.syn_order:
                idx = self.syn_order.index(col)
                self.syn_order.insert(idx, cat_col)

            self.col_map[cat_col] = {
                "original_dtype": "category",
                "converted_type": "category",
                "method": self.col_map[col]["method"]
            }
        return X

    def _apply_converted_dtype(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.syn_order:
            if col not in X.columns:
                continue
            cinfo = self.col_map[col]
            ctype = cinfo["converted_type"]
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

    def _update_varsel_dict_after_split(self, X: pd.DataFrame) -> None:
        new_splits = {}
        for col in self.syn_order:
            if col.endswith("_cat"):
                base = col[:-4]
                if base in self.col_map:
                    new_splits[base] = col

        # 새 col_cat이 아직 varsel에 없으면 기본 predictor
        for col in X.columns:
            if col not in self.variable_selection_ and col in self.syn_order:
                idx = self.syn_order.index(col)
                self.variable_selection_[col] = self.syn_order[:idx]

        # base_col이 predictor인 곳 => cat_col도 추가
        for tgt_col, preds in self.variable_selection_.items():
            updated = set(preds)
            for bcol, ccol in new_splits.items():
                if bcol in updated:
                    updated.add(ccol)
            self.variable_selection_[tgt_col] = list(updated)

    # ----------------------
    def get_info(self) -> Dict[str, Any]:
        """
        고정된 키 이름들로 반환
        special_value => {col: [ ... ]}
        """
        orig_dtype_map = {}
        conv_type_map = {}
        method_map = {}
        for c, info in self.col_map.items():
            orig_dtype_map[c] = info.get("original_dtype")
            conv_type_map[c] = info.get("converted_type")
            method_map[c] = info.get("method")

        return {
            "syn_order": self.syn_order,
            "original_dtype": orig_dtype_map,
            "converted_type": conv_type_map,
            "method": method_map,
            # 사용자 + auto합쳐진 special values
            "special_value": self.special_value,  # {col: [ ... ]}
            "date_mins": self.date_mins,
            "variable_selection": self.variable_selection_,
        }
