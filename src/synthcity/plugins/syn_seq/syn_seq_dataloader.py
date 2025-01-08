# File: plugins/syn_seq/syn_seq_dataloader.py

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader, Constraints
from .syn_seq_encoder import Syn_SeqEncoder


class Syn_SeqDataLoader(DataLoader):
    """
    A DataLoader that applies Syn_Seq-style preprocessing to input data,
    inheriting directly from DataLoader and implementing all required
    abstract methods.

    - target_order: the order of columns to keep or process
    - columns_special_values: map of { column_name : special_value(s) }
    - unique_value_threshold: used to decide numeric vs. categorical
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_order: List[str],
        columns_special_values: Optional[Dict[str, Any]] = None,
        unique_value_threshold: int = 20,
        random_state: int = 0,
        train_size: float = 0.8,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Syn_SeqDataLoader with preprocessing parameters and data.

        Args:
            data (pd.DataFrame): The input DataFrame.
            target_order (List[str]): Columns to retain and process in specific order.
            columns_special_values (Optional[Dict[str, Any]]): Mapping of columns to special values.
            unique_value_threshold (int): Threshold to classify columns as numeric or categorical.
            random_state (int): For reproducibility in train/test splits, etc.
            train_size (float): Ratio for train/test.
            **kwargs: Additional arguments for base DataLoader.
        """
        # 검증
        missing_columns = set(target_order) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in input data: {missing_columns}")

        self.target_order = target_order
        self.columns_special_values = columns_special_values or {}
        self.unique_value_threshold = unique_value_threshold

        # 순서대로 컬럼 정렬
        filtered_data = data[self.target_order].copy()

        # 부모 DataLoader 생성자 호출
        super().__init__(
            data_type="syn_seq",
            data=filtered_data,
            random_state=random_state,
            train_size=train_size,
            **kwargs,
        )

        # 최종 내부 보관용 DF
        self._df = filtered_data

    # ----------------------------------------------------------------------
    # DataLoader에서 요구하는 추상 메서드들 구현
    # ----------------------------------------------------------------------

    @property
    def shape(self) -> tuple:
        return self._df.shape

    @property
    def columns(self) -> list:
        return list(self._df.columns)

    def dataframe(self) -> pd.DataFrame:
        return self._df

    def numpy(self) -> np.ndarray:
        return self._df.values

    def info(self) -> dict:
        return {
            "data_type": self.data_type,
            "len": len(self),
            "train_size": self.train_size,
            "random_state": self.random_state,
            "target_order": self.target_order,
            "unique_value_threshold": self.unique_value_threshold,
            # 필요 시 추가 필드
        }

    def __len__(self) -> int:
        return len(self._df)

    def satisfies(self, constraints: Constraints) -> bool:
        # 예: constraints가 self._df에 모두 만족하는지
        return constraints.is_valid(self._df)

    def match(self, constraints: Constraints) -> "Syn_SeqDataLoader":
        # 예: constraints를 만족하도록 row/col을 필터링한 DF
        matched_df = constraints.match(self._df)
        return self.decorate(matched_df)

    @staticmethod
    def from_info(data: pd.DataFrame, info: dict) -> "Syn_SeqDataLoader":
        # info 딕셔너리를 사용하여 동일한 파라미터로 재생성
        return Syn_SeqDataLoader(
            data=data,
            target_order=info["target_order"],
            unique_value_threshold=info["unique_value_threshold"],
            random_state=info["random_state"],
            train_size=info["train_size"],
        )

    def sample(self, count: int, random_state: int = 0) -> "Syn_SeqDataLoader":
        sampled_df = self._df.sample(count, random_state=random_state)
        return self.decorate(sampled_df)

    def drop(self, columns: list = []) -> "Syn_SeqDataLoader":
        dropped_df = self._df.drop(columns=columns, errors="ignore")
        return self.decorate(dropped_df)

    def __getitem__(self, feature: Union[str, list, int]) -> Any:
        return self._df[feature]

    def __setitem__(self, feature: str, val: Any) -> None:
        self._df[feature] = val

    def train(self) -> "Syn_SeqDataLoader":
        # 간단한 예: random split (stratify=None)
        # 실제론 sklearn train_test_split 등을 활용
        ntrain = int(len(self._df) * self.train_size)
        train_df = self._df.iloc[:ntrain].copy()
        return self.decorate(train_df)

    def test(self) -> "Syn_SeqDataLoader":
        ntrain = int(len(self._df) * self.train_size)
        test_df = self._df.iloc[ntrain:].copy()
        return self.decorate(test_df)

    def fillna(self, value: Any) -> "Syn_SeqDataLoader":
        filled_df = self._df.fillna(value)
        return self.decorate(filled_df)

    def compression_protected_features(self) -> list:
        # 예시로 아무것도 보호하지 않는다고 가정
        return []

    def is_tabular(self) -> bool:
        # syn_seq는 일반 tabular 데이터 가정
        return True

    def unpack(self, as_numpy: bool = False, pad: bool = False) -> Any:
        """
        Roughly, if you want to separate X, y, etc. For now we do a simple pass:
        """
        if as_numpy:
            return self._df.to_numpy()
        return self._df

    def get_fairness_column(self) -> Union[str, Any]:
        # 필요하다면, 예: 'sex' 컬럼등 반환
        return None

    # ----------------------------------------------------------------------
    # encode/decode : Syn_SeqEncoder 사용 예시
    # ----------------------------------------------------------------------

    def encode(
        self, encoders: Optional[Dict[str, Any]] = None
    ) -> Tuple["Syn_SeqDataLoader", Dict]:
        if encoders is None:
            encoder = Syn_SeqEncoder(
                columns_special_values=self.columns_special_values,
                target_order=self.target_order,
                unique_value_threshold=self.unique_value_threshold,
            )
            encoder.fit(self._df)
            encoded_data = encoder.transform(self._df)

            new_loader = self.decorate(encoded_data)
            return new_loader, {"syn_seq_encoder": encoder}
        else:
            # 만약 이미 encoders가 있다면, 상위 로직 또는 사용자 정의로 처리
            # 예: for col, enc in encoders.items(): ...
            return self, encoders  # 예시

    def decode(
        self, encoders: Dict[str, Any]
    ) -> "Syn_SeqDataLoader":
        if "syn_seq_encoder" in encoders:
            encoder = encoders["syn_seq_encoder"]
            if not isinstance(encoder, Syn_SeqEncoder):
                raise TypeError(f"Expected Syn_SeqEncoder, got {type(encoder)}")

            decoded_data = encoder.inverse_transform(self._df)
            return self.decorate(decoded_data)
        else:
            return self

    # ----------------------------------------------------------------------
    # decorate : 내부 DF 교체 + 동일한 설정 유지
    # ----------------------------------------------------------------------
    def decorate(self, data: pd.DataFrame) -> "Syn_SeqDataLoader":
        """
        Build a new Syn_SeqDataLoader with the same configurations,
        but a new underlying dataframe.
        """
        return Syn_SeqDataLoader(
            data=data,
            target_order=self.target_order,
            columns_special_values=self.columns_special_values,
            unique_value_threshold=self.unique_value_threshold,
            random_state=self.random_state,
            train_size=self.train_size,
        )
