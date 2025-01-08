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

    - syn_order: the order of columns to keep or process
    - columns_special_values: map of { column_name : special_value(s) }
    - max_categories: used to decide numeric vs. categorical
    """

    def __init__(
        self,
        data: pd.DataFrame,
        syn_order: List[str],
        columns_special_values: Optional[Dict[str, Any]] = None,
        max_categories: int = 20,
        random_state: int = 0,
        train_size: float = 0.8,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Syn_SeqDataLoader with preprocessing parameters and data.

        Args:
            data (pd.DataFrame): The input DataFrame.
            syn_order (List[str]): Columns to retain and process in specific order.
            columns_special_values (Optional[Dict[str, Any]]): Mapping of columns to special values.
            max_categories (int): Threshold to classify columns as numeric or categorical.
            random_state (int): For reproducibility in train/test splits, etc.
            train_size (float): Ratio for train/test.
            **kwargs: Additional arguments for base DataLoader.
        """
        # 검증
        missing_columns = set(syn_order) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in input data: {missing_columns}")

        self.syn_order = syn_order
        self.columns_special_values = columns_special_values or {}
        self.max_categories = max_categories

        # 순서대로 컬럼 정렬
        filtered_data = data[self.syn_order].copy()

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

        # ---- 1) 여기서 encoder.fit()까지만 수행 ----
        self._encoder = Syn_SeqEncoder(
            columns_special_values=self.columns_special_values,
            syn_order=self.syn_order,
            max_categories=self.max_categories,
        )
        # fit까지는 여기서!
        self._encoder.fit(self._df)
        # transform은 encode() 시점까지 미룬다.

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
            "syn_order": self.syn_order,
            "max_categories": self.max_categories,
            # 필요 시 추가 필드
        }

    def __len__(self) -> int:
        return len(self._df)

    def satisfies(self, constraints: Constraints) -> bool:
        return constraints.is_valid(self._df)

    def match(self, constraints: Constraints) -> "Syn_SeqDataLoader":
        matched_df = constraints.match(self._df)
        return self.decorate(matched_df)

    @staticmethod
    def from_info(data: pd.DataFrame, info: dict) -> "Syn_SeqDataLoader":
        return Syn_SeqDataLoader(
            data=data,
            syn_order=info["syn_order"],
            max_categories=info["max_categories"],
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
        return []

    def is_tabular(self) -> bool:
        return True

    def unpack(self, as_numpy: bool = False, pad: bool = False) -> Any:
        if as_numpy:
            return self._df.to_numpy()
        return self._df

    def get_fairness_column(self) -> Union[str, Any]:
        return None

    # ----------------------------------------------------------------------
    # encode/decode : encoder.transform or inverse_transform
    # ----------------------------------------------------------------------

    def encode(
        self, encoders: Optional[Dict[str, Any]] = None
    ) -> Tuple["Syn_SeqDataLoader", Dict]:
        """
        Called typically by a plugin's fit(...).
        At this point, we want to do 'transform' using self._encoder 
        that was already fitted in __init__.
        """
        if encoders is None:
            # self._encoder는 이미 __init__에서 fit까지 끝냄
            encoded_data = self._encoder.transform(self._df)

            new_loader = self.decorate(encoded_data)
            # encoders dict에 저장해서 plugin에서 활용할 수 있게
            return new_loader, {"syn_seq_encoder": self._encoder}
        else:
            # 이미 encoders가 있다면, 굳이 transform 하지 않고 그대로 반환
            # (혹은 encoders로 새 transform을 해도 된다. 목적에 따라 다름.)
            return self, encoders

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

    def decorate(self, data: pd.DataFrame) -> "Syn_SeqDataLoader":
        return Syn_SeqDataLoader(
            data=data,
            syn_order=self.syn_order,
            columns_special_values=self.columns_special_values,
            max_categories=self.max_categories,
            random_state=self.random_state,
            train_size=self.train_size,
        )
