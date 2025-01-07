# file: synthcity/plugins/syn_seq/methods/syn_norm.py

from typing import Any, Dict
import pandas as pd
import numpy as np
import statsmodels.api as sm


def syn_norm_fit(X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
    """
    y ~ X 선형회귀(정규) param fit
    """
    df = pd.concat([X, y], axis=1).dropna()
    if df.empty:
        return {"model": None}

    # design matrix
    X_ = sm.add_constant(df[X.columns], has_constant="add")
    y_ = df[y.name]

    model = sm.OLS(y_, X_).fit()
    return {"model": model}


def syn_norm_generate(model_params: Dict[str, Any], X: pd.DataFrame, count: int = None, **kwargs):
    """
    OLS 계수를 사용해 mean + residual variance 로부터 샘플
    """
    if model_params["model"] is None:
        if count is None:
            count = len(X)
        return [np.nan]*count

    model = model_params["model"]
    # design matrix
    X_ = sm.add_constant(X, has_constant="add")

    # 예측값
    mean = model.predict(X_)

    # 잔차 분산
    sigma = np.sqrt(model.mse_resid)
    syn_y = np.random.normal(loc=mean, scale=sigma, size=len(X_))

    return syn_y
