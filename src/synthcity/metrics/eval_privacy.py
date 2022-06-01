# stdlib
from collections import Counter
from typing import Any, Dict

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# synthcity absolute
from synthcity.metrics._utils import get_features
from synthcity.plugins.core.dataloader import DataLoader

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
    def evaluate_data(self, X: DataLoader) -> int:

        features = get_features(X, X.sensitive_features)

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
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        return {
            "gt": self.evaluate_data(X_gt),
            "syn": (self.evaluate_data(X_syn) + 1e-8),
        }


class lDiversityDistinct(PrivacyEvaluator):
    """Returns the distinct l-diversity ratio between the real data and the synthetic data.

    For each dataset, it computes the minimum value l which satisfies the distinct l-diversity rule: every generalized block has to contain at least l different sensitive values.

    We simulate a set of the cluster over the dataset, and we return the minimum length of unique sensitive values for any cluster.
    """

    @staticmethod
    def name() -> str:
        return "distinct l-diversity"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def evaluate_data(self, X: DataLoader) -> int:
        features = get_features(X, X.sensitive_features)

        values = [999]
        for n_clusters in [2, 5, 10, 15]:
            if len(X) / n_clusters < 10:
                continue
            model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0).fit(
                X[features]
            )
            clusters = model.predict(X.dataframe()[features])
            clusters_df = pd.Series(clusters, index=X.dataframe().index)
            for cluster in range(n_clusters):
                partition = X.dataframe()[clusters_df == cluster]
                uniq_values = partition[X.sensitive_features].drop_duplicates()
                values.append(len(uniq_values))

        return int(np.min(values))

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        return {
            "gt": self.evaluate_data(X_gt),
            "syn": (self.evaluate_data(X_syn) + 1e-8),
        }


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
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        features = get_features(X_gt, X_gt.sensitive_features)

        values = []
        for n_clusters in [2, 5, 10, 15]:
            if len(X_gt) / n_clusters < 10:
                continue
            model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0).fit(
                X_gt[features]
            )
            clusters = model.predict(X_syn[features])
            counts: dict = Counter(clusters)
            values.append(np.min(list(counts.values())))

        if len(values) == 0:
            return {"score": 0}

        return {"score": int(np.min(values))}


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
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        features = get_features(X_gt, X_gt.sensitive_features)

        values = []
        for n_clusters in [2, 5, 10, 15]:
            if len(X_gt) / n_clusters < 10:
                continue
            model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0).fit(
                X_gt[features]
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

        return {"score": float(np.max(values))}


class IdentifiabilityScore(PrivacyEvaluator):
    """Returns the maximum re-identification probability on the real dataset from the synthetic dataset.

    For each dataset partition, we report the maximum ratio of unique sensitive information between the real dataset and in the synthetic dataset.
    """

    @staticmethod
    def name() -> str:
        return "identifiability_score"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        """Compare Wasserstein distance between original data and synthetic data.

        Args:
            orig_data: original data
            synth_data: synthetically generated data

        Returns:
            WD_value: Wasserstein distance
        """

        # Entropy computation
        def compute_entropy(labels: np.ndarray) -> np.ndarray:
            value, counts = np.unique(np.round(labels), return_counts=True)
            return entropy(counts)

        # Parameters
        no, x_dim = X_gt.shape

        # Weights
        W = np.zeros(
            [
                x_dim,
            ]
        )

        for i in range(x_dim):
            W[i] = compute_entropy(X_gt.numpy()[:, i])

        # Normalization
        X_hat = X_gt.numpy().copy()
        X_syn_hat = X_syn.numpy().copy()

        eps = 1e-16
        W = np.ones_like(W)

        for i in range(x_dim):
            X_hat[:, i] = X_gt.numpy()[:, i] * 1.0 / (W[i] + eps)
            X_syn_hat[:, i] = X_syn.numpy()[:, i] * 1.0 / (W[i] + eps)

        # r_i computation
        nbrs = NearestNeighbors(n_neighbors=2).fit(X_hat)
        distance, _ = nbrs.kneighbors(X_hat)

        # hat{r_i} computation
        nbrs_hat = NearestNeighbors(n_neighbors=1).fit(X_syn_hat)
        distance_hat, _ = nbrs_hat.kneighbors(X_hat)

        # See which one is bigger
        R_Diff = distance_hat[:, 0] - distance[:, 1]
        identifiability_value = np.sum(R_Diff < 0) / float(no)

        return {"score": identifiability_value}
