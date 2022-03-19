# third party
import pandas as pd
from pydantic import validate_arguments
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def select_outliers(
    X_gt: pd.DataFrame, y_gt: pd.Series, method: str = "local_outlier_factor"
) -> pd.Index:
    X_gt["target"] = y_gt

    if method == "isolation_forests":
        predictions = IsolationForest().fit_predict(X_gt)
    elif method == "local_outlier_factor":
        predictions = LocalOutlierFactor().fit_predict(X_gt)
    elif method == "elliptic_envelope":
        predictions = EllipticEnvelope().fit_predict(X_gt)
    else:
        raise RuntimeError(f"Unknown outlier method {method}")

    outliers = pd.Series(predictions, index=X_gt.index)
    return outliers == -1


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def select_quantiles(
    X_gt: pd.DataFrame,
    quantiles: int = 5,
) -> pd.DataFrame:
    X = X_gt.copy()
    for col in X.columns:
        if len(X[col].unique()) > quantiles:
            X[col] = pd.qcut(X[col], quantiles, labels=False, duplicates="drop")

    return X


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_k_anonimity(X: pd.DataFrame, k: int) -> bool:
    Xq = select_quantiles(X, quantiles=5)
    Xq_groupby = Xq.groupby(list(X.columns)).size().reset_index(name="count")

    return bool(Xq_groupby["count"].min() >= k)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def attack_predict_real_data_outliers(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_synth: pd.DataFrame, y_synth: pd.Series
) -> None:
    pass
