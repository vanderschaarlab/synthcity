# file: synthcity/plugins/syn_seq/methods/syn_cart.py

from typing import Any, Dict
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import numpy as np


def syn_cart_fit(X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
    """
    CART fitting for a numeric or categorical y.
    Returns model parameters (in this case, a DecisionTree + ydata).
    """
    # 간단히 y가 numeric인지 factor인지 판별
    is_numeric = pd.api.types.is_numeric_dtype(y)
    if is_numeric:
        # regression tree
        tree = DecisionTreeRegressor(random_state=kwargs.get("random_state", 0))
    else:
        # classification tree
        tree = DecisionTreeClassifier(random_state=kwargs.get("random_state", 0))

    # missing or NA 제거 (필요시)
    df = pd.concat([X, y], axis=1).dropna()
    X_clean = df[X.columns]
    y_clean = df[y.name]

    if len(X_clean) == 0:
        # edge case
        return {"model": None, "is_numeric": is_numeric, "leaf_data": {}}

    tree.fit(X_clean, y_clean)

    # leaf별 원본 y를 저장 => generate할 때 random pick
    leaf_data = {}
    leaves = tree.apply(X_clean)
    for leaf_id in np.unique(leaves):
        # 해당 leaf에 속한 y 값들
        leaf_data[leaf_id] = y_clean[leaves == leaf_id].values.tolist()

    return {
        "model": tree,
        "is_numeric": is_numeric,
        "leaf_data": leaf_data
    }


def syn_cart_generate(model_params: Dict[str, Any], X: pd.DataFrame, count: int = None, **kwargs):
    """
    Using fitted CART model, generate synthetic y
    """
    if model_params["model"] is None:
        # edge case
        # e.g. fill with NaN or some default
        if count is None:
            count = len(X)
        return [np.nan]*count

    tree = model_params["model"]
    leaf_data = model_params["leaf_data"]

    if count is None:
        # 그냥 X의 행 개수만큼
        count = len(X)

    # predict leaf
    leaves_pred = tree.apply(X)

    # for each row -> leaf -> random pick from leaf_data
    syn_y = []
    for i in range(len(X)):
        leaf_id = leaves_pred[i]
        candidates = leaf_data.get(leaf_id, [])
        if len(candidates) == 0:
            syn_y.append(np.nan)
        else:
            choice = np.random.choice(candidates)
            syn_y.append(choice)

    # 만약 X가 count보다 적다면, row를 추가로 샘플링하는 방식을 쓸 수도 있지만
    # 여기서는 단순하게 len(X)개만 만든다고 가정
    return syn_y
