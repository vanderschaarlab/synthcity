# stdlib
import platform
from abc import abstractmethod
from collections import Counter
from typing import Any, Dict, Optional

# third party
import domias
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments
from scipy import stats
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# synthcity absolute
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
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        ...

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{X_gt.hash()}_{X_syn.hash()}_{self._reduction}_{platform.python_version()}.bkp"
        )
        if self.use_cache(cache_file):
            return load_from_file(cache_file)

        results = self._evaluate(X_gt, X_syn)
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


class DomiasScores(PrivacyEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_privacy.domias
        :parts: 1

    A dictionary with a key for each of the `synthetic_sizes` values.
    For each `synthetic_sizes` value, the dictionary contains the keys:
        * `MIA_performance` : accuracy and AUCROC for each attack
        * `MIA_scores`: output scores for each attack
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="syn", **kwargs)

    @staticmethod
    def name() -> str:
        return "domias"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        results = self._evaluate_performance(X_gt, X_syn)

        oc_results = self._evaluate_performance(X_gt, X_syn)

        for key in oc_results:
            results[key] = oc_results[key]

        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate_performance(
        generator: domias.models.generator.GeneratorInterface,
        dataset: np.ndarray,
        mem_set_size: int,
        reference_set_size: int,
        training_epochs: int = 2000,
        synthetic_sizes: list = [10000],
        density_estimator: str = "prior",
        seed: int = 0,
        device: Any = DEVICE,
        shifted_column: Optional[int] = None,
        zero_quantile: float = 0.3,
        reference_kept_p: float = 1.0,
    ) -> Dict:
        """
        Evaluate various Membership Inference Attacks, using the `generator` and the `dataset`.
        The provided generator must not be fitted.

        Args:
            generator: GeneratorInterface
                Generator with the `fit` and `generate` methods. The generator MUST not be fitted.
            dataset: int
                The evaluation dataset, used to derive the training and test datasets.
            training_size: int
                The split for the training (member) dataset out of `dataset`
            reference_size: int
                The split for the reference dataset out of `dataset`.
            training_epochs: int
                Training epochs
            synthetic_sizes: List[int]
                For how many synthetic samples to test the attacks.
            density_estimator: str, default = "prior"
                Which density to use. Available options:
                    * prior
                    * bnaf
                    * kde
            seed: int
                Random seed
            device: PyTorch device
                CPU or CUDA
            shifted_column: Optional[int]
                Shift a column
            zero_quantile: float
                Threshold for shifting the column.
            reference_kept_p: float
                Reference dataset parameter (for distributional shift experiment)

        Returns:
            A dictionary with a key for each of the `synthetic_sizes` values.
            For each `synthetic_sizes` value, the dictionary contains the keys:
                * `MIA_performance` : accuracy and AUCROC for each attack
                * `MIA_scores`: output scores for each attack
                * `data`: the evaluation data
            For both `MIA_performance` and `MIA_scores`, the following attacks are evaluated:
                * "ablated_eq1"
                * "ablated_eq2"
                * "LOGAN_D1"
                * "MC"
                * "gan_leaks"
                * "gan_leaks_cal"
                * "LOGAN_0"
                * "eq1"
                * "domias"
        """
        performance_logger: Dict = {}

        continuous = []
        for i in np.arange(dataset.shape[1]):
            if len(np.unique(dataset[:, i])) < 10:
                continuous.append(0)
            else:
                continuous.append(1)

        norm = domias.evaluator.normal_func_feat(dataset, continuous)

        # For experiment with domain shift in reference dataset
        if shifted_column is not None:
            thres = np.quantile(dataset[:, shifted_column], zero_quantile) + 0.01
            dataset[:, shifted_column][dataset[:, shifted_column] < thres] = -999.0
            dataset[:, shifted_column][dataset[:, shifted_column] > thres] = 999.0
            dataset[:, shifted_column][dataset[:, shifted_column] == -999.0] = 0.0
            dataset[:, shifted_column][dataset[:, shifted_column] == 999.0] = 1.0

            mem_set = dataset[:mem_set_size]  # membership set
            mem_set = mem_set[mem_set[:, shifted_column] == 1]

            non_mem_set = dataset[mem_set_size : 2 * mem_set_size]  # set of non-members
            non_mem_set = non_mem_set[: len(mem_set)]
            reference_set = dataset[-reference_set_size:]

            # Used for experiment with distributional shift in reference dataset
            reference_set_A1 = reference_set[reference_set[:, shifted_column] == 1]
            reference_set_A0 = reference_set[reference_set[:, shifted_column] == 0]
            reference_set_A0_kept = reference_set_A0[
                : int(len(reference_set_A0) * reference_kept_p)
            ]
            if reference_kept_p > 0:
                reference_set = np.concatenate(
                    (reference_set_A1, reference_set_A0_kept), 0
                )
            else:
                reference_set = reference_set_A1
                # non_mem_set = non_mem_set_A1

            mem_set_size = len(mem_set)
            reference_set_size = len(reference_set)

            # hide column A
            mem_set = np.delete(mem_set, shifted_column, 1)
            non_mem_set = np.delete(non_mem_set, shifted_column, 1)
            reference_set = np.delete(reference_set, shifted_column, 1)
            dataset = np.delete(dataset, shifted_column, 1)
        # For all other experiments
        else:
            mem_set = dataset[:mem_set_size]
            non_mem_set = dataset[mem_set_size : 2 * mem_set_size]
            reference_set = dataset[-reference_set_size:]

        """ 3. Synthesis with the GeneratorInferface"""
        df = pd.DataFrame(mem_set)
        df.columns = [str(_) for _ in range(dataset.shape[1])]

        # Train generator
        generator.fit(df)

        for synthetic_size in synthetic_sizes:
            performance_logger[synthetic_size] = {
                "MIA_performance": {},
                "MIA_scores": {},
                "data": {},
            }
            synth_set = generator.generate(synthetic_size)
            synth_val_set = generator.generate(synthetic_size)

            wd_n = min(len(synth_set), len(reference_set))
            eval_met_on_reference = domias.metrics.wd.compute_wd(
                synth_set[:wd_n], reference_set[:wd_n]
            )
            performance_logger[synthetic_size]["MIA_performance"][
                "sample_quality"
            ] = eval_met_on_reference

            # get real test sets of members and non members
            X_test = np.concatenate([mem_set, non_mem_set])
            Y_test = np.concatenate(
                [np.ones(mem_set.shape[0]), np.zeros(non_mem_set.shape[0])]
            ).astype(bool)

            performance_logger[synthetic_size]["data"]["Xtest"] = X_test
            performance_logger[synthetic_size]["data"]["Ytest"] = Y_test

            """ 4. density estimation / evaluation of Eqn.(1) & Eqn.(2)"""
            # First, estimate density of synthetic data
            # BNAF for pG
            if density_estimator == "bnaf":
                _, p_G_model = domias.bnaf.density_estimator.density_estimator_trainer(
                    synth_set.values,
                    synth_val_set.values[: int(0.5 * synthetic_size)],
                    synth_val_set.values[int(0.5 * synthetic_size) :],
                )
                _, p_R_model = domias.bnaf.density_estimator.density_estimator_trainer(
                    reference_set
                )
                p_G_evaluated = np.exp(
                    domias.bnaf.density_estimator.compute_log_p_x(
                        p_G_model, torch.as_tensor(X_test).float().to(device)
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )

            # KDE for pG
            elif density_estimator == "kde":
                density_gen = stats.gaussian_kde(synth_set.values.transpose(1, 0))
                density_data = stats.gaussian_kde(reference_set.transpose(1, 0))
                p_G_evaluated = density_gen(X_test.transpose(1, 0))
            elif density_estimator == "prior":
                density_gen = stats.gaussian_kde(synth_set.values.transpose(1, 0))
                density_data = stats.gaussian_kde(reference_set.transpose(1, 0))
                p_G_evaluated = density_gen(X_test.transpose(1, 0))

            # eqn2: \prop P_G(x_i)/P_X(x_i)
            # DOMIAS (BNAF for p_R estimation)
            if density_estimator == "bnaf":
                p_R_evaluated = np.exp(
                    domias.bnaf.density_estimator.compute_log_p_x(
                        p_R_model, torch.as_tensor(X_test).float().to(device)
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )

            # DOMIAS (KDE for p_R estimation)
            elif density_estimator == "kde":
                p_R_evaluated = density_data(X_test.transpose(1, 0))

            # DOMIAS (with prior for p_R, see Appendix experiment)
            elif density_estimator == "prior":
                p_R_evaluated = norm.pdf(X_test)

            p_rel = p_G_evaluated / (p_R_evaluated + 1e-10)

            acc, auc = domias.baselines.compute_metrics_baseline(p_rel, Y_test)
            performance_logger[synthetic_size]["MIA_performance"]["domias"] = {
                "accuracy": acc,
                "aucroc": auc,
            }

            performance_logger[synthetic_size]["MIA_scores"]["domias"] = p_rel

        return performance_logger
