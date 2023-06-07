# stdlib
import platform
from abc import abstractmethod
from collections import Counter
from typing import Any, Dict, Tuple, Union

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments
from scipy import stats
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# synthcity absolute
import synthcity.logger as log
from synthcity.metrics import _utils
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.utils.constants import DEVICE
from synthcity.utils.serialization import load_from_file, save_to_file

# synthcity relative
from .core import MetricEvaluator


class PrivacyEvaluator(MetricEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_privacy.PrivacyEvaluator
        :parts: 1
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        return "privacy"

    @abstractmethod
    def _evaluate(
        self, X_gt: DataLoader, X_syn: DataLoader, *args: Any, **kwargs: Any
    ) -> Dict:
        ...

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self, X_gt: DataLoader, X_syn: DataLoader, *args: Any, **kwargs: Any
    ) -> Dict:
        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{X_gt.hash()}_{X_syn.hash()}_{self._reduction}_{platform.python_version()}.bkp"
        )
        if self.use_cache(cache_file):
            return load_from_file(cache_file)
        results = self._evaluate(X_gt, X_syn, *args, **kwargs)
        save_to_file(cache_file, results)
        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_default(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> float:
        return self.evaluate(X_gt, X_syn)[self._default_metric]


class kAnonymization(PrivacyEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_privacy.kAnonymization
        :parts: 1

    Returns the k-anon ratio between the real data and the synthetic data.
    For each dataset, it is computed the value k which satisfies the k-anonymity rule: each record is similar to at least another k-1 other records on the potentially identifying variables.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="syn", **kwargs)

    @staticmethod
    def name() -> str:
        return "k-anonymization"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_data(self, X: DataLoader) -> int:
        features = _utils.get_features(X, X.sensitive_features)

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
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        if X_gt.type() == "images":
            raise ValueError(f"Metric {self.name()} doesn't support images")

        return {
            "gt": self.evaluate_data(X_gt),
            "syn": (self.evaluate_data(X_syn) + 1e-8),
        }


class lDiversityDistinct(PrivacyEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_privacy.lDiversityDistinct
        :parts: 1

    Returns the distinct l-diversity ratio between the real data and the synthetic data.

    For each dataset, it computes the minimum value l which satisfies the distinct l-diversity rule: every generalized block has to contain at least l different sensitive values.

    We simulate a set of the cluster over the dataset, and we return the minimum length of unique sensitive values for any cluster.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="syn", **kwargs)

    @staticmethod
    def name() -> str:
        return "distinct l-diversity"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def evaluate_data(self, X: DataLoader) -> int:
        features = _utils.get_features(X, X.sensitive_features)

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
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        if X_gt.type() == "images":
            raise ValueError(f"Metric {self.name()} doesn't support images")

        return {
            "gt": self.evaluate_data(X_gt),
            "syn": (self.evaluate_data(X_syn) + 1e-8),
        }


class kMap(PrivacyEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_privacy.kMap
        :parts: 1

    Returns the minimum value k that satisfies the k-map rule.

    The data satisfies k-map if every combination of values for the quasi-identifiers appears at least k times in the reidentification(synthetic) dataset.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)

    @staticmethod
    def name() -> str:
        return "k-map"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        if X_gt.type() == "images":
            raise ValueError(f"Metric {self.name()} doesn't support images")

        features = _utils.get_features(X_gt, X_gt.sensitive_features)

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
    """
    .. inheritance-diagram:: synthcity.metrics.eval_privacy.DeltaPresence
        :parts: 1

    Returns the maximum re-identification probability on the real dataset from the synthetic dataset.

    For each dataset partition, we report the maximum ratio of unique sensitive information between the real dataset and in the synthetic dataset.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)

    @staticmethod
    def name() -> str:
        return "delta-presence"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        if X_gt.type() == "images":
            raise ValueError(f"Metric {self.name()} doesn't support images")

        features = _utils.get_features(X_gt, X_gt.sensitive_features)

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
    """
    .. inheritance-diagram:: synthcity.metrics.eval_privacy.IdentifiabilityScore
        :parts: 1

    Returns the re-identification score on the real dataset from the synthetic dataset.

    We estimate the risk of re-identifying any real data point using synthetic data.
    Intuitively, if the synthetic data are very close to the real data, the re-identification risk would be high.
    The precise formulation of the re-identification score is given in the reference below.

    Reference: Jinsung Yoon, Lydia N. Drumright, Mihaela van der Schaar,
    "Anonymization through Data Synthesis using Generative Adversarial Networks (ADS-GAN):
    A harmonizing advancement for AI in medicine,"
    IEEE Journal of Biomedical and Health Informatics (JBHI), 2019.
    Paper link: https://ieeexplore.ieee.org/document/9034117
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="score_OC", **kwargs)

    @staticmethod
    def name() -> str:
        return "identifiability_score"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        results = self._compute_scores(X_gt, X_syn)

        oc_results = self._compute_scores(X_gt, X_syn, "OC")

        for key in oc_results:
            results[key] = oc_results[key]
        log.info("ID_score results: ", results)
        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _compute_scores(
        self, X_gt: DataLoader, X_syn: DataLoader, emb: str = ""
    ) -> Dict:
        """Compare Wasserstein distance between original data and synthetic data.

        Args:
            orig_data: original data
            synth_data: synthetically generated data

        Returns:
            WD_value: Wasserstein distance
        """
        X_gt_ = X_gt.numpy().reshape(len(X_gt), -1)
        X_syn_ = X_syn.numpy().reshape(len(X_syn), -1)

        if emb == "OC":
            emb = f"_{emb}"
            oneclass_model = self._get_oneclass_model(X_gt_)
            X_gt_ = self._oneclass_predict(oneclass_model, X_gt_)
            X_syn_ = self._oneclass_predict(oneclass_model, X_syn_)
        else:
            if emb != "":
                raise RuntimeError(f" Invalid emb {emb}")

        # Entropy computation
        def compute_entropy(labels: np.ndarray) -> np.ndarray:
            value, counts = np.unique(np.round(labels), return_counts=True)
            return entropy(counts)

        # Parameters
        no, x_dim = X_gt_.shape

        # Weights
        W = np.zeros(
            [
                x_dim,
            ]
        )

        for i in range(x_dim):
            W[i] = compute_entropy(X_gt_[:, i])

        # Normalization
        X_hat = X_gt_
        X_syn_hat = X_syn_

        eps = 1e-16
        W = np.ones_like(W)

        for i in range(x_dim):
            X_hat[:, i] = X_gt_[:, i] * 1.0 / (W[i] + eps)
            X_syn_hat[:, i] = X_syn_[:, i] * 1.0 / (W[i] + eps)

        # r_i computation
        nbrs = NearestNeighbors(n_neighbors=2).fit(X_hat)
        distance, _ = nbrs.kneighbors(X_hat)

        # hat{r_i} computation
        nbrs_hat = NearestNeighbors(n_neighbors=1).fit(X_syn_hat)
        distance_hat, _ = nbrs_hat.kneighbors(X_hat)

        # See which one is bigger
        R_Diff = distance_hat[:, 0] - distance[:, 1]
        identifiability_value = np.sum(R_Diff < 0) / float(no)

        return {f"score{emb}": identifiability_value}


class DomiasMIA(PrivacyEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_privacy.domias
        :parts: 1

    DOMIAS is a membership inference attacker model against synthetic data, that incorporates
    density estimation to detect generative model overfitting. That is it uses local overfitting to
    detect whether a data point was used to train the generative model or not.

    Returns:
    A dictionary with a key for each of the `synthetic_sizes` values.
    For each `synthetic_sizes` value, the dictionary contains the keys:
        * `MIA_performance` : accuracy and AUCROC for each attack
        * `MIA_scores`: output scores for each attack

    Reference: Boris van Breugel, Hao Sun, Zhaozhi Qian,  Mihaela van der Schaar, AISTATS 2023.
    DOMIAS: Membership Inference Attacks against Synthetic Data through Overfitting Detection.

    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="aucroc", **kwargs)

    @staticmethod
    def name() -> str:
        return "DomiasMIA"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_default(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
        X_train: DataLoader,
        X_ref_syn: DataLoader,
        reference_size: int,
    ) -> float:
        return self.evaluate(
            X_gt,
            X_syn,
            X_train,
            X_ref_syn,
            reference_size=reference_size,
        )[self._default_metric]

    @abstractmethod
    def evaluate_p_R(
        self,
        synth_set: Union[DataLoader, Any],
        synth_val_set: Union[DataLoader, Any],
        reference_set: np.ndarray,
        X_test: np.ndarray,
        device: Any,
    ) -> Any:
        ...

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X_gt: Union[
            DataLoader, Any
        ],  # TODO: X_gt needs to be big enough that it can be split into non_mem_set and also ref_set
        synth_set: Union[DataLoader, Any],
        X_train: Union[DataLoader, Any],
        synth_val_set: Union[DataLoader, Any],
        reference_size: int = 100,  # look at default sizes
        device: Any = DEVICE,
    ) -> Dict:
        """
        Evaluate various Membership Inference Attacks, using the `generator` and the `dataset`.
        The provided generator must not be fitted.

        Args:
            generator: GeneratorInterface
                Generator with the `fit` and `generate` methods. The generator MUST not be fitted.
            X_gt: Union[DataLoader, Any]
                The evaluation dataset, used to derive the training and test datasets.
            synth_set: Union[DataLoader, Any]
                The synthetic dataset.
            X_train: Union[DataLoader, Any]
                The dataset used to create the mem_set.
            synth_val_set: Union[DataLoader, Any]
                The dataset used to calculate the density of the synthetic data
            reference_size: int
                The size of the reference dataset
            device: PyTorch device
                CPU or CUDA

        Returns:
            A dictionary with the AUCROC and accuracy scores for the attack.
        """

        mem_set = X_train.dataframe()
        non_mem_set, reference_set = (
            X_gt.numpy()[:reference_size],
            X_gt.numpy()[-reference_size:],
        )

        all_real_data = np.concatenate((X_train.numpy(), X_gt.numpy()), axis=0)

        continuous = []
        for i in np.arange(all_real_data.shape[1]):
            if len(np.unique(all_real_data[:, i])) < 10:
                continuous.append(0)
            else:
                continuous.append(1)

        self.norm = _utils.normal_func_feat(all_real_data, continuous)

        """ 3. Synthesis with the GeneratorInferface"""

        # get real test sets of members and non members
        X_test = np.concatenate([mem_set, non_mem_set])
        Y_test = np.concatenate(
            [np.ones(mem_set.shape[0]), np.zeros(non_mem_set.shape[0])]
        ).astype(bool)

        """ 4. density estimation / evaluation of Eqn.(1) & Eqn.(2)"""
        # First, estimate density of synthetic data then
        # eqn2: \prop P_G(x_i)/P_X(x_i)
        # p_R estimation
        p_G_evaluated, p_R_evaluated = self.evaluate_p_R(
            synth_set, synth_val_set, reference_set, X_test, device
        )

        p_rel = p_G_evaluated / (p_R_evaluated + 1e-10)

        acc, auc = _utils.compute_metrics_baseline(p_rel, Y_test)
        return {
            "accuracy": acc,
            "aucroc": auc,
        }


class DomiasMIAPrior(DomiasMIA):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "DomiasMIA_prior"

    def evaluate_p_R(
        self,
        synth_set: Union[DataLoader, Any],
        synth_val_set: Union[DataLoader, Any],
        reference_set: np.ndarray,
        X_test: np.ndarray,
        device: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        density_gen = stats.gaussian_kde(synth_set.values.transpose(1, 0))
        p_G_evaluated = density_gen(X_test.transpose(1, 0))
        p_R_evaluated = self.norm.pdf(X_test)
        return p_G_evaluated, p_R_evaluated


class DomiasMIAKDE(DomiasMIA):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "DomiasMIA_KDE"

    def evaluate_p_R(
        self,
        synth_set: Union[DataLoader, Any],
        synth_val_set: Union[DataLoader, Any],
        reference_set: np.ndarray,
        X_test: np.ndarray,
        device: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if synth_set.shape[0] > X_test.shape[0]:
            log.critical(
                """
The data appears to lie in a lower-dimensional subspace of the space in which it is expressed.
This has resulted in a singular data covariance matrix, which cannot be treated using the algorithms
implemented in `gaussian_kde`. If you wish to use the density estimator `kde` or `prior`, consider performing principle component analysis / dimensionality reduction
and using `gaussian_kde` with the transformed data. Else consider using `bnaf` as the density estimator.
                """
            )

        density_gen = stats.gaussian_kde(synth_set.values.transpose(1, 0))
        density_data = stats.gaussian_kde(reference_set.transpose(1, 0))
        p_G_evaluated = density_gen(X_test.transpose(1, 0))
        p_R_evaluated = density_data(X_test.transpose(1, 0))
        return p_G_evaluated, p_R_evaluated


class DomiasMIABNAF(DomiasMIA):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "DomiasMIA_BNAF"

    def evaluate_p_R(
        self,
        synth_set: Union[DataLoader, Any],
        synth_val_set: Union[DataLoader, Any],
        reference_set: np.ndarray,
        X_test: np.ndarray,
        device: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, p_G_model = _utils.density_estimator_trainer(
            synth_set.values,
            synth_val_set.values[: int(0.5 * synth_val_set.shape[0])],
            synth_val_set.values[int(0.5 * synth_val_set.shape[0]) :],
        )
        _, p_R_model = _utils.density_estimator_trainer(reference_set)
        p_G_evaluated = np.exp(
            _utils.compute_log_p_x(
                p_G_model, torch.as_tensor(X_test).float().to(device)
            )
            .cpu()
            .detach()
            .numpy()
        )
        p_R_evaluated = np.exp(
            _utils.compute_log_p_x(
                p_R_model, torch.as_tensor(X_test).float().to(device)
            )
            .cpu()
            .detach()
            .numpy()
        )
        return p_G_evaluated, p_R_evaluated
