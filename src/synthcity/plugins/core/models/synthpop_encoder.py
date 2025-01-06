from typing import Optional, Dict, Any, List
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np

# SynthpopEncoder
class SynthpopEncoder(TransformerMixin, BaseEstimator):
    """
    SynthpopEncoder handles preprocessing and postprocessing tasks using fit/transform pattern.
    """

    def __init__(
        self,
        columns_special_values: Optional[Dict[str, Any]] = None,
        target_order: Optional[List[str]] = None,
        unique_value_threshold: int = 20,
    ) -> None:
        self.columns_special_values = columns_special_values or {}
        self.target_order = target_order or []
        self.unique_value_threshold = unique_value_threshold

        self.categorical_info_ = {}
        self.numeric_info_ = {}
        self.column_order_ = None
        self.method_assignments = {}
        self.prediction_matrix = None

    def fit(self, X: pd.DataFrame, y=None) -> "SynthpopEncoder":
        """학습 단계에서 필요한 변환 정보를 수집"""
        X = X.copy()

        # 1. 컬럼 순서 저장
        self.column_order_ = (
            self.target_order if self.target_order else X.columns.tolist()
        )

        # 2. 유니크 값 개수를 기준으로 컬럼 타입 결정
        for col in X.columns:
            unique_values = X[col].nunique()

            if unique_values > self.unique_value_threshold:  # Numeric 기준
                self.numeric_info_[col] = {"dtype": X[col].dtype}
            else:
                self.categorical_info_[col] = {"dtype": X[col].dtype}

        # 3. 고빈도 값 바로 특수 값에 추가
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            high_freq_vals = freq[freq > 0.9].index.tolist()
            if high_freq_vals:
                self.columns_special_values[col] = (
                    self.columns_special_values.get(col, []) + high_freq_vals
                )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """학습된 정보를 사용하여 데이터 변환"""
        X = X.copy()
        X = self._reorder_columns(X)  # 1. 컬럼 순서 정렬
        X = self._categorize_numeric(X)  # 2. Numeric 컬럼의 Categorization
        X = self._update_column_types(X)  # 3. 컬럼 dtype 변경
        X = self._assign_methods(X)  # 4. 컬럼별 method 지정 리스트 생성
        X = self._generate_prediction_matrix(X)  # 5. Prediction Matrix 생성
        return X

    def _reorder_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """컬럼 순서를 fit 단계에서 저장한 순서로 정렬"""
        if self.column_order_:
            return X[self.column_order_]
        return X

    def _categorize_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Numeric 컬럼을 카테고리로 분리.
        기존 컬럼에서 special_value와 NaN을 분리하고,
        새로운 컬럼에 카테고리로 저장.
        """
        for col in self.numeric_info_:
            if col in X.columns:
                new_col = f"{col}_cat"

                special_values = self.columns_special_values.get(col, [])
                X[new_col] = X[col].apply(
                    lambda x: x if x in special_values or pd.isna(x) else -777
                )

                X[new_col] = X[new_col].fillna(-9999)

                X[col] = X[col].apply(
                    lambda x: x if x not in special_values and not pd.isna(x) else pd.NA
                )

                X[new_col] = X[new_col].astype("category")

        return X

    def _update_column_types(self, X: pd.DataFrame) -> pd.DataFrame:
        """각 컬럼마다 dtype 변경"""
        for col in X.columns:
            if col in self.numeric_info_:
                X[col] = X[col].astype(self.numeric_info_[col]["dtype"])
            elif col in self.categorical_info_:
                X[col] = X[col].astype(self.categorical_info_[col]["dtype"])
        return X

    def _assign_methods(self, X: pd.DataFrame) -> pd.DataFrame:
        """각 컬럼마다 method를 지정하는 리스트 생성"""
        first_col = True
        for col in X.columns:
            if first_col:
                self.method_assignments[col] = "random_sampling"
                first_col = False
            else:
                self.method_assignments[col] = "CART"
        return X

    def _generate_prediction_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prediction Matrix 생성"""
        self.prediction_matrix = pd.DataFrame(index=X.index, columns=X.columns)
        for col in X.columns:
            self.prediction_matrix[col] = 0
        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """변환된 데이터를 원래 형태로 복원"""
        X = X.copy()

        # 특수값 복원
        for col, special_val in self.columns_special_values.items():
            if col in X.columns:
                X[col] = X[col].replace(pd.NA, special_val)

        # 데이터 타입 복원
        for col, info in self.categorical_info_.items():
            if col in X.columns:
                X[col] = X[col].astype(info["dtype"])

        for col, info in self.numeric_info_.items():
            if col in X.columns:
                X[col] = X[col].astype(info["dtype"])

        return X
