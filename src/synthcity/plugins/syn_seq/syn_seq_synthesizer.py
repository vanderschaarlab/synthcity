# file: synthcity/plugins/syn_seq/syn_seq_synthesizer.py

from typing import Any, Dict, List, Optional
import pandas as pd

# 예시: 이미 작성된 encoder
from .syn_seq_encoder import Syn_SeqEncoder

# 예시: method별 합성 함수
from .methods.syn_cart import syn_cart_fit, syn_cart_generate
from .methods.syn_norm import syn_norm_fit, syn_norm_generate
# 필요에 따라 추가 import
# from .methods.syn_logreg import syn_logreg_fit, syn_logreg_generate
# ...

# (옵션) rules
from .rules.rules import apply_rules


class SynSeqSynthesizer:
    """
    R-synthpop 처럼:
      - visit.sequence에 따라 각 변수별로 model fit -> generate
      - 최종 합성 데이터를 반환
    """

    def __init__(
        self,
        visit_sequence: Optional[List[str]] = None,
        method_map: Optional[Dict[str, str]] = None,
        encoder: Optional[Syn_SeqEncoder] = None,
        use_rules: bool = False,
        **kwargs
    ):
        """
        Args:
            visit_sequence: e.g. ["col1", "col2", ...] 순서
            method_map: 각 변수별 사용할 합성 method. e.g. {"col1":"cart", "col2":"norm", ...}
            encoder: 데이터 인코더
            use_rules: restricted values 등을 적용할지 여부
        """
        self.visit_sequence = visit_sequence or []
        self.method_map = method_map or {}  # {varName: "cart"/"norm"/...}
        self.encoder = encoder if encoder else Syn_SeqEncoder()
        self.use_rules = use_rules
        self.is_fitted = False

        # 각 변수별로 학습된 파라미터(회귀계수, 트리 등)를 저장
        self.fitted_models: Dict[str, Any] = {}
        self.original_data: Optional[pd.DataFrame] = None

    def fit(self, data: pd.DataFrame, *args, **kwargs) -> "SynSeqSynthesizer":
        """
        1) encoder로 transform(전처리)
        2) visit_sequence 순서대로 각 변수를 학습.
        """
        self.original_data = data.copy()

        # 1) encoder
        enc_data = self.encoder.fit_transform(self.original_data)

        # 2) visit_sequence에서 하나씩 합성모델 fit
        #    (단, R-synthpop와 달리 여기서는 fit만, generate는 별도함수에서)
        #    (혹은 fit할때 바로 generate해서 대체해도 됨)
        for var in self.visit_sequence:
            if var not in enc_data.columns:
                raise ValueError(f"Variable {var} not found in data.")

            method = self.method_map.get(var, "cart")  # default "cart"
            # predictor는 "이미 합성된 / or 원본" X. 
            # 여기서는 단순히 enc_data.drop(var, axis=1)을 predictor로 사용 가능.
            Xpred = enc_data.drop(columns=[var])
            y = enc_data[var]

            if method == "cart":
                model_params = syn_cart_fit(Xpred, y, **kwargs)
            elif method == "norm":
                model_params = syn_norm_fit(Xpred, y, **kwargs)
            else:
                raise NotImplementedError(f"Unknown method {method}")

            self.fitted_models[var] = (method, model_params)

            # (선택) partial generation 후, enc_data[var]를 새로 대체
            #        R-synthpop는 var 합성 -> 결과를 데이터셋에 반영 -> 다음 변수로
            #        Python도 가능. 아래는 "fit단계에서 generate" 로직.
            #        (원본 R-synthpop 스타일을 충실히 재현하려면 여기에 generate가 들어가고,
            #         그 결과로 enc_data[var]를 업데이트)
            # syn_y = self._generate_for_var(var, Xpred, count=len(Xpred))
            # enc_data[var] = syn_y

        self.is_fitted = True
        return self

    def generate(self, count: Optional[int] = None, *args, **kwargs) -> pd.DataFrame:
        """
        visit_sequence 순서대로, 학습된 모델을 사용해서 synthetic data를 생성
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit before generate")

        if self.original_data is None:
            raise RuntimeError("No original data stored. fit() not run?")

        enc_data = self.encoder.transform(self.original_data)

        # 만약 count가 None이면, 원본 행 개수로 합성
        if count is None:
            count = len(enc_data)

        # 빈 DF에 predictor만들기 or Xpred shape(count, ?)
        # 여기서는 원본 enc_data.columns를 기반으로 (count)개 row를 만들 수도 있고,
        # 또는 predictor는 random sampling
        synthetic_df = pd.DataFrame(index=range(count), columns=enc_data.columns)
        synthetic_df = synthetic_df.fillna(0)  # 임시. 실제론 predictor별 처리

        # R-synthpop와 유사하게, var 별로 순회
        for var in self.visit_sequence:
            method, model_params = self.fitted_models[var]
            # predictor X: synthetic_df.drop(columns=[var])
            Xpred = synthetic_df.drop(columns=[var])

            if method == "cart":
                syn_y = syn_cart_generate(model_params, Xpred, **kwargs)
            elif method == "norm":
                syn_y = syn_norm_generate(model_params, Xpred, **kwargs)
            else:
                raise NotImplementedError(f"Unknown method {method}")

            # update synthetic
            synthetic_df[var] = syn_y

            # (옵션) rules 적용
            if self.use_rules:
                synthetic_df = apply_rules(synthetic_df, var)

        # encoder.inverse_transform
        synthetic_dec = self.encoder.inverse_transform(synthetic_df)
        return synthetic_dec

    def _generate_for_var(self, var: str, Xpred: pd.DataFrame, count: int, *args, **kwargs):
        """
        (선택) fit 단계에서 partial generate 시에 사용 가능
        """
        method, model_params = self.fitted_models[var]
        if method == "cart":
            syn_y = syn_cart_generate(model_params, Xpred, count=count, **kwargs)
        elif method == "norm":
            syn_y = syn_norm_generate(model_params, Xpred, count=count, **kwargs)
        else:
            raise NotImplementedError(f"Unknown method {method}")
        return syn_y

