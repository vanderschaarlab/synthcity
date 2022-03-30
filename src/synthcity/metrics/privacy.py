# stdlib
from collections import Counter
from typing import Any

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

# synthcity relative
from .core import MetricEvaluator


class PrivacyEvaluator(MetricEvaluator):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        return "privacy"


class kAnonymization(PrivacyEvaluator):
    """Returns the k-anon ratio between the real data and the syhnthetic data.
    For each dataset, it is computed the value k which satisfies the k-anonymity rule: each record is similar to at least another k-1 other records on the potentially identifying variables.
    """

    @staticmethod
    def name() -> str:
        return "k-anonymization"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate_data(self, X: pd.DataFrame) -> int:

        features = get_features(X, self._sensitive_columns)

        values = [999]
        for n_clusters in [2, 5, 10, 15]:
            if len(X) / n_clusters < 10:
                continue
            cluster = KMeans(
                n_clusters=n_clusters, init="k-means++", random_state=0
            ).fit(X[features])
            counts: dict = Counter(cluster.labels_)
            values.append(np.min(list(counts.values())))

        return int(np.min(values))

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X: pd.DataFrame, X_syn: pd.DataFrame) -> float:
        return self._evaluate_data(X) / (self._evaluate_data(X_syn) + 1e-8)


class lDiversity(PrivacyEvaluator):
    """Returns the l-diversity ratio between the real data and the synthetic data.

    For each dataset, it computes the minimum value l which satisfies the l-diversity rule: every generalized block has to contain at least l different sensitive values.

    We simulate a set of the cluster over the dataset, and we return the minimum length of unique sensitive values for any cluster.
    """

    @staticmethod
    def name() -> str:
        return "l-diversity"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def _evaluate_data(self, X: pd.DataFrame) -> int:
        features = get_features(X, self._sensitive_columns)

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
                uniq_values = partition[self._sensitive_columns].drop_duplicates()
                values.append(len(uniq_values))

        return int(np.min(values))

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X: pd.DataFrame, X_syn: pd.DataFrame) -> float:
        return self._evaluate_data(X) / (self._evaluate_data(X_syn) + 1e-8)


class kMap(PrivacyEvaluator):
    """Returns the minimum value k that satisfies the k-map rule.

    The data satisfies k-map if every combination of values for the quasi-identifiers appears at least k times in the reidentification(synthetic) dataset.
    """

    @staticmethod
    def name() -> str:
        return "k-map"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X: pd.DataFrame, X_syn: pd.DataFrame) -> float:
        features = get_features(X, self._sensitive_columns)

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


class DeltaPresence(PrivacyEvaluator):
    """Returns the maximum re-identification probability on the real dataset from the synthetic dataset.

    For each dataset partition, we report the maximum ratio of unique sensitive information between the real dataset and in the synthetic dataset.
    """

    @staticmethod
    def name() -> str:
        return "delta-presence"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X: pd.DataFrame, X_syn: pd.DataFrame) -> float:
        features = get_features(X, self._sensitive_columns)

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
