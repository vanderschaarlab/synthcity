from typing import Optional, Dict, List, Any
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Syn_SeqEncoder(TransformerMixin, BaseEstimator):
    """
    syn_seq용 최소화된 인코더.
      - 전처리(날짜 변환, special_value 등)는 이미 preprocess.py에서 처리했다고 가정.
      - 여기서는 syn_order, method, variable_selection만 세팅.
    """

    def __init__(
        self,
        syn_order: Optional[List[str]] = None,
        method: Optional[Dict[str, str]] = None,
        variable_selection: Optional[Dict[str, List[str]]] = None,
        default_method: str = "cart",
    ):
        """
        Args:
            syn_order: 유저가 지정한 컬럼 순서. 비어있으면 prepare 단계에서 X.columns 사용
            method: {col: "rf"/"norm"/...} 식으로 유저 지정. 
            variable_selection: {col: [predictors...]}
            default_method: 유저 지정 없을 때 쓸 기본값. (ex: "cart")
        """
        self.syn_order = syn_order or []
        self.method = method or {}
        self.variable_selection_ = variable_selection or {}
        self.default_method = default_method

        # 내부 관리용
        self.col_map: Dict[str, Dict[str, Any]] = {}  # {col: {"method": "..."}}

    def prepare(self, X: pd.DataFrame) -> "Syn_SeqEncoder":
        """
        1) syn_order 확정
        2) assign_method_to_cols
        3) variable_selection 설정
        """
        self._set_syn_order(X)
        self._assign_method_to_cols()
        self._assign_variable_selection()
        return self

    def _set_syn_order(self, X: pd.DataFrame):
        """
        syn_order가 비어 있으면 X.columns를 사용,
        유효하지 않은 컬럼(존재X)은 제외
        """
        if not self.syn_order:
            self.syn_order = list(X.columns)
        else:
            self.syn_order = [c for c in self.syn_order if c in X.columns]

    def _assign_method_to_cols(self):
        """
        1) 첫 컬럼은 user 지정 없으면 swr
        2) 나머지 컬럼은 user 지정 없으면 default_method
        """
        self.col_map.clear()
        for i, col in enumerate(self.syn_order):
            user_m = self.method.get(col)
            if i == 0:
                chosen = user_m if user_m else "swr"
            else:
                chosen = user_m if user_m else self.default_method

            self.col_map[col] = {"method": chosen}

    def _assign_variable_selection(self):
        """
        유저가 variable_selection을 지정하지 않은 컬럼은,
        (현재 컬럼 인덱스보다 앞선 컬럼들)을 predictor로 설정
        """
        for i, col in enumerate(self.syn_order):
            if col not in self.variable_selection_:
                self.variable_selection_[col] = self.syn_order[:i]

    def get_info(self) -> Dict[str, Any]:
        """
        Encoder가 관리 중인 syn_order, method, variable_selection 등 반환
        """
        method_map = {}
        for col in self.col_map:
            method_map[col] = self.col_map[col]["method"]

        return {
            "syn_order": self.syn_order,
            "method": method_map,
            "variable_selection": self.variable_selection_,
        }
