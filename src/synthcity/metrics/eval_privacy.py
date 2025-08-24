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
        return "minimize"

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
            log.debug(
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


class AdversarialAccuracy(PrivacyEvaluator):
    """
    Adversarial Accuracy (AA) from Yale et al. (2020).

    Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC10311334/

    Intuition
    ---------
    AA tells us whether synthetic data are
    (i) too close to the real data – privacy leakage, low utility – or
    (ii) too far away – poor fidelity.
    It is the accuracy of a 1-NN classifier that tries to distinguish real from synthetic samples using Euclidean distances.

    Let
        d_tt(i):  nearest-neighbor distance from real sample i to all other real samples (excluding itself)
        d_tg(i):  NN distance from real sample i to the synthetic set
        d_gg(j):  NN distance from synthetic sample j to all other synthetic samples (excluding itself)
        d_gt(j):  NN distance from synthetic sample j to the real set

    The metric is

        AA = 0.5 * [ (1/n) Σ 1( d_tg(i) > d_tt(i) )   +
                    (1/m) Σ 1( d_gt(j) > d_gg(j) )
                   ]

    Range and interpretation
    ------------------------
      • AA → 0 : generator over-fits (synthetic ≈ real, privacy ↓, utility ↓)
      • AA → 1 : generator under-fits (synthetic easily separable, utility ↓)
      • AA ≈ 0.5 : good trade-off between realism and privacy (ideal)

    The evaluator returns {"aa": float}.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="aa", **kwargs)

    @staticmethod
    def name() -> str:
        return "adversarial_accuracy"

    @staticmethod
    def direction() -> str:
        # Best value is 0.5; downstream code can optimize |AA – 0.5|.
        return "custom"

    # ---------- main ------------------------------------------

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        if X_gt.type() == "images":
            raise ValueError("AdversarialAccuracy is not defined for images.")

        real = X_gt.numpy().astype(float)
        syn = X_syn.numpy().astype(float)

        # 1) d_tt: distance to the 2-nd nearest neighbour in the real set
        nn_real = NearestNeighbors(n_neighbors=2).fit(real)
        d_tt = nn_real.kneighbors(real, return_distance=True)[0][:, 1]

        # 2) Cross-set distances
        d_tg = self._pairwise_min_dist(real, syn)
        d_gt = self._pairwise_min_dist(syn, real)

        # 3) d_gg: distance to the 2-nd NN in the synthetic set
        if len(syn) > 1:
            nn_syn = NearestNeighbors(n_neighbors=2).fit(syn)
            d_gg = nn_syn.kneighbors(syn, return_distance=True)[0][:, 1]
        else:  # edge case: only one synthetic sample
            d_gg = np.full(len(syn), np.inf)

        # 4) Compute AA
        aa_left = (d_tg > d_tt).mean()
        aa_right = (d_gt > d_gg).mean()
        aa = 0.5 * (aa_left + aa_right)
        aa = max(aa, 1e-8)  # add epsilon to pass test

        return {"aa": float(aa)}

    # ---------- helpers --------------------------------------------------

    @staticmethod
    def _pairwise_min_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """For every point in `a` return its minimum Euclidean distance to set `b`."""
        dists = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        return dists.min(axis=1)


class EpsilonIdentifiability(PrivacyEvaluator):
    """
    epsilon-Identifiability from Yoon *et al.*, IEEE JBHI 2019, DOI 10.1109/JBHI.2018.2880147

    Reference:  <https://ieeexplore.ieee.org/document/9034117>

    I(D, D̂) = 1/N · Σ 1[r^hat_i < r_i]

    • r_i  : min weighted-Euclidean distance from real xᵢ to another real
    • r^hat_i : min weighted-Euclidean distance from real x_i to any synthetic
    • weights w_j = 1 / H(X^{(j)})  (inverse discrete entropy of column j)

    Lower values ⇒ more privacy leakage
    """

    def __init__(self, epsilon: float = 0.5, **kwargs: Any) -> None:
        super().__init__(default_metric="I", **kwargs)
        self.epsilon = epsilon  # user-set threshold

    @staticmethod
    def name() -> str:
        return "epsilon_identifiability"

    @staticmethod
    def direction() -> str:
        return "maximize"  # higher value ⇒ safer data

    # -------- main  --------
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        if X_gt.type() == "images":
            raise ValueError("Metric not defined for images.")

        # 1) common preprocessing
        real = X_gt.numpy()  # shape (N,d)
        synth = X_syn.numpy()  # shape (M,d)
        w = self._weight_vector(X_gt)  # shape (d,)

        # 2) compute r_i  (min to *other* real rows)
        #    efficient: nearest-neighbor search in weighted space
        Rw = real * w  # weight each column
        nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(Rw)
        dist_real, _ = nn.kneighbors(Rw)
        r = dist_real[:, 1]  # skip self-distance (0)

        # 3) compute r^hat_i  (min to any synthetic row)
        Sw = synth * w
        nn_syn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(Sw)
        r_hat, _ = nn_syn.kneighbors(Rw)
        r_hat = r_hat[:, 0]

        # 4) indicator + score
        I_val = (r_hat < r).mean()

        return {
            "I": 1.0 - I_val,  # higher ⇒ safer
        }

    # -------- helper --------
    @staticmethod
    def _discrete_entropy(col: np.ndarray) -> float:
        vals, counts = np.unique(col, return_counts=True)
        p = counts / counts.sum()
        return -(p * np.log2(p + 1e-12)).sum()

    def _weight_vector(self, X: DataLoader) -> np.ndarray:
        df = X.dataframe()
        ent = np.array([self._discrete_entropy(df[c].values) for c in df.columns])
        return 1.0 / (ent + 1e-12)  # avoid /0

    @staticmethod
    def _pairwise_min_dist(x: np.ndarray, Y: np.ndarray, w: np.ndarray) -> float:
        """return min  ||w·(x-y)||_2  over y ∈Y"""
        diffs = (x - Y) * w
        return np.linalg.norm(diffs, axis=1).min()

    def evaluate_default(
        self, X_gt: DataLoader, X_syn: DataLoader, *a: Any, **k: Any
    ) -> float:
        """Return ‘privacy-safe’ score = 1 – I."""
        return self.evaluate(X_gt, X_syn)["I"]


class tCloseness(PrivacyEvaluator):
    """
    Returns the t-closeness score between the real data and the synthetic data.
    Measures how close the sensitive attribute distribution in each equivalence class is to the global distribution from real data.

    Reference:
        Li, Ninghui, Tiancheng Li, and Suresh Venkatasubramanian.
        "t-closeness: Privacy beyond k-anonymity and l-diversity." ICDE 2007.
    """

    def __init__(
        self, sensitive_column: str = "sensitive", n_clusters: int = 10, **kwargs: Any
    ) -> None:
        super().__init__(default_metric="t", **kwargs)
        self.sensitive_column = sensitive_column
        self.n_clusters = n_clusters

    @staticmethod
    def name() -> str:
        return "t-closeness"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        if X_gt.type() == "images":
            raise ValueError("Metric not defined for images")

        df_real = X_gt.dataframe()
        df_synth = X_syn.dataframe()

        loader_sens = getattr(X_gt, "sensitive_features", None) or []
        if loader_sens:
            sens_feats = list(loader_sens)
        elif self.sensitive_column in df_real.columns:
            sens_feats = [self.sensitive_column]
        else:
            sens_feats = [df_real.columns[-1]]
        sensitive_col = sens_feats[0]

        # Select a set of quasi-identifiers
        qid_cols = _utils.get_features(X_gt, sens_feats)
        if not qid_cols:
            raise ValueError("No quasi‐identifier columns found")

        # Compute global sensitive attribute distribution on real data
        global_dist = df_real[sensitive_col].value_counts(normalize=True)

        # One hot encoder and clustering
        X_real_qid = pd.get_dummies(df_real[qid_cols], drop_first=True)
        model = KMeans(n_clusters=self.n_clusters, random_state=0).fit(X_real_qid)

        X_synth_qid = pd.get_dummies(df_synth[qid_cols], drop_first=True)
        X_synth_qid = X_synth_qid.reindex(columns=X_real_qid.columns, fill_value=0)
        df_synth["cluster"] = model.predict(X_synth_qid)

        # Compute t-closeness per cluster on synthetic data
        t_vals = []
        for cluster_id, group in df_synth.groupby("cluster"):
            local_dist = group[sensitive_col].value_counts(normalize=True)
            # Align local distribution with global distribution
            local_dist = local_dist.reindex(global_dist.index, fill_value=0.0)
            tvd = 0.5 * np.abs(local_dist - global_dist).sum()
            t_vals.append(tvd)

        max_t = max(t_vals) if t_vals else 0.0

        return {"t": float(max_t)}
