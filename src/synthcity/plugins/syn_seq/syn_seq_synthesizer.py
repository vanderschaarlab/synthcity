# file: synthcity/models/syn_seq/syn_seq_synthesizer.py

from typing import Any, Optional
import pandas as pd

# 예: 이미 작성하셨다고 가정하는 encoder
from synthcity.syn_seq_encoder import SynSeqEncoder

# methods/*.py 에 포팅된 함수들(예: syn_cart, syn_norm 등)
# from synthcity.models.syn_seq.methods.syn_cart import syn_cart
# from synthcity.models.syn_seq.methods.syn_norm import syn_norm
# ...

class SynSeqSynthesizer:
    """
    - R-synthpop 로직을 기반으로 한 (fit, generate) 등을 제공
    - 여러 method를 조합하여 full synthetic data를 생성
    """

    def __init__(self, encoder: Optional[SynSeqEncoder] = None):
        self.encoder = encoder if encoder is not None else SynSeqEncoder()
        self.is_fitted = False

        # 필요시 각 변수별 학습결과 저장용
        self.models = {}

    def fit(self, X: pd.DataFrame, *args, **kwargs) -> "SynSeqSynthesizer":
        """
        R-synthpop처럼 각 변수별(visit.sequence 순서)로 CART나 norm 등을 적용,
        학습(회귀계수, 트리 등) 결과를 self.models에 저장
        """
        # 1) Encoder로 변환 (카테고리, 결측치 처리 등)
        X_enc = self.encoder.fit_transform(X)

        # 2) 예시: 단순히 모든 변수를 CART로 학습한다고 가정
        # for col in X_enc.columns:
        #     model_info = syn_cart_fit(X_enc[col], ...)
        #     self.models[col] = model_info

        self.is_fitted = True
        return self

    def generate(self, count: int = None, *args, **kwargs) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before generate().")

        # count 지정 없으면 원본 행 수와 동일하게 할 수도 있음
        # if count is None:
        #     count = self.original_nrows (예: fit에서 저장)

        # 1) 변수별 synthetic 값 생성
        # synthetic_dict = {}
        # for col in X_enc.columns:
        #     syn_vals = syn_cart_generate(self.models[col], count, ...)
        #     synthetic_dict[col] = syn_vals

        # syn_data = pd.DataFrame(synthetic_dict)

        # 2) encoder 역변환
        # syn_data_dec = self.encoder.inverse_transform(syn_data)
        # return syn_data_dec

        # 여기서는 예시로 간단히
        df_fake = pd.DataFrame({
            "colA": [1,2,3],
            "colB": [9,8,7]
        })
        return df_fake
