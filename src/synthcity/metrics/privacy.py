# stdlib
from collections import Counter
from typing import List

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# synthcity absolute
from synthcity.metrics._utils import get_features


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def select_outliers(
    X_gt: pd.DataFrame, method: str = "local_outlier_factor"
) -> pd.Index:
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
def evaluate_k_anonymization(X: pd.DataFrame, sensitive_columns: List[str] = []) -> int:
    """Returns the minimum value k which satisfies the k-anonymity rule: each record is similar to at least another k-1 other records on the potentially identifying variables.

    We simulate a set of clusters over the dataset and return the minimum length of a cluster, from all trials.
    """
    features = get_features(X, sensitive_columns)

    values = [999]
    for n_clusters in [2, 5, 10, 15]:
        if len(X) / n_clusters < 10:
            continue
        cluster = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0).fit(
            X[features]
        )
        counts: dict = Counter(cluster.labels_)
        values.append(np.min(list(counts.values())))

    return int(np.min(values))


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_l_diversity(X: pd.DataFrame, sensitive_columns: List[str]) -> int:
    """Returns the minimum value l which satisfies the l-diversity rule: every generalized block has to contain at least l different sensitive values.

    We simulate a set of the cluster over the dataset, and we return the minimum length of unique sensitive values for any cluster.

    """
    features = get_features(X, sensitive_columns)

    values = [999]
    for n_clusters in [2, 5, 10, 15]:
        if len(X) / n_clusters < 10:
            continue
        model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0).fit(
            X[features]
        )
        clusters = model.predict(X[features])
        clusters_df = pd.Series(clusters, index=X.index)
        for cluster in range(n_clusters):
            partition = X[clusters_df == cluster]
            uniq_values = partition[sensitive_columns].drop_duplicates()
            values.append(len(uniq_values))

    return int(np.min(values))


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_kmap(
    X: pd.DataFrame, X_syn: pd.DataFrame, sensitive_columns: List[str]
) -> int:
    """Returns the minimum value k that satisfies the k-map rule.

    The data satisfies k-map if every combination of values for the quasi-identifiers appears at least k times in the reidentification(synthetic) dataset.
    """
    features = get_features(X, sensitive_columns)

    values = []
    for n_clusters in [2, 5, 10, 15]:
        if len(X) / n_clusters < 10:
            continue
        model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0).fit(
            X[features]
        )
        clusters = model.predict(X_syn[features])
        counts: dict = Counter(clusters)
        values.append(np.min(list(counts.values())))

    if len(values) == 0:
        return 0

    return int(np.min(values))


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_delta_presence(
    X: pd.DataFrame, X_syn: pd.DataFrame, sensitive_columns: List[str]
) -> float:
    """Returns the maximum re-identification probability on the real dataset from the synthetic dataset.

    For each dataset partition, we report the maximum ratio of unique sensitive information between the real dataset and in the synthetic dataset.

    """
    features = get_features(X, sensitive_columns)

    values = []
    for n_clusters in [2, 5, 10, 15]:
        if len(X) / n_clusters < 10:
            continue
        model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0).fit(
            X[features]
        )
        clusters = model.predict(X_syn[features])
        synth_counts: dict = Counter(clusters)
        gt_counts: dict = Counter(model.labels_)

        for key in gt_counts:
            if key not in synth_counts:
                continue
            gt_cnt = gt_counts[key]
            synth_cnt = synth_counts[key]

            delta = gt_cnt / (synth_cnt + 1e-8)

            values.append(delta)

    return float(np.max(values))
