# stdlib
import platform
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

# third party
import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss
from pydantic import validate_arguments
from scipy import linalg
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
from scipy.stats import chisquare, ks_2samp
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# synthcity absolute
import synthcity.logger as log
from synthcity.metrics._utils import get_frequency
from synthcity.metrics.core import MetricEvaluator
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.models.survival_analysis.metrics import (
    nonparametric_distance,
)
from synthcity.utils.reproducibility import clear_cache
from synthcity.utils.serialization import load_from_file, save_to_file


class StatisticalEvaluator(MetricEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.StatisticalEvaluator
        :parts: 1

    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        return "stats"

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

        clear_cache()
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


class InverseKLDivergence(StatisticalEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.InverseKLDivergence
        :parts: 1


    Returns the average inverse of the Kullback–Leibler Divergence metric.

    Score:
        0: the datasets are from different distributions.
        1: the datasets are from the same distribution.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="marginal", **kwargs)

    @staticmethod
    def name() -> str:
        return "inv_kl_divergence"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        freqs = get_frequency(
            X_gt.dataframe(), X_syn.dataframe(), n_histogram_bins=self._n_histogram_bins
        )
        res = []
        for col in X_gt.columns:
            gt_freq, synth_freq = freqs[col]
            res.append(1 / (1 + np.sum(kl_div(gt_freq, synth_freq))))

        return {"marginal": float(self.reduction()(res))}


class KolmogorovSmirnovTest(StatisticalEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.KolmogorovSmirnovTest
        :parts: 1

    Performs the Kolmogorov-Smirnov test for goodness of fit.

    Score:
        0: the distributions are totally different.
        1: the distributions are identical.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="marginal", **kwargs)

    @staticmethod
    def name() -> str:
        return "ks_test"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        res = []
        for col in X_gt.columns:
            statistic, _ = ks_2samp(X_gt[col], X_syn[col])
            res.append(1 - statistic)

        return {"marginal": float(self.reduction()(res))}


class ChiSquaredTest(StatisticalEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.ChiSquaredTest
        :parts: 1

    Performs the one-way chi-square test.

    Returns:
        The p-value. A small value indicates that we can reject the null hypothesis and that the distributions are different.

    Score:
        0: the distributions are different
        1: the distributions are identical.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="marginal", **kwargs)

    @staticmethod
    def name() -> str:
        return "chi_squared_test"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        res = []
        freqs = get_frequency(
            X_gt.dataframe(), X_syn.dataframe(), n_histogram_bins=self._n_histogram_bins
        )

        for col in X_gt.columns:
            gt_freq, synth_freq = freqs[col]
            try:
                _, pvalue = chisquare(gt_freq, synth_freq)
                if np.isnan(pvalue):
                    pvalue = 0
            except BaseException:
                log.error("chisquare failed")
                pvalue = 0

            res.append(pvalue)

        return {"marginal": float(self.reduction()(res))}


class MaximumMeanDiscrepancy(StatisticalEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.MaximumMeanDiscrepancy
        :parts: 1

    Empirical maximum mean discrepancy. The lower the result the more evidence that distributions are the same.

    Args:
        kernel: "rbf", "linear" or "polynomial"

    Score:
        0: The distributions are the same.
        1: The distributions are totally different.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(self, kernel: str = "rbf", **kwargs: Any) -> None:
        super().__init__(default_metric="joint", **kwargs)

        self.kernel = kernel

    @staticmethod
    def name() -> str:
        return "max_mean_discrepancy"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        if self.kernel == "linear":
            """
            MMD using linear kernel (i.e., k(x,y) = <x,y>)
            """
            delta_df = X_gt.dataframe().mean(axis=0) - X_syn.dataframe().mean(axis=0)
            delta = delta_df.values

            score = delta.dot(delta.T)
        elif self.kernel == "rbf":
            """
            MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
            """
            gamma = 1.0
            XX = metrics.pairwise.rbf_kernel(
                X_gt.numpy().reshape(len(X_gt), -1),
                X_gt.numpy().reshape(len(X_gt), -1),
                gamma,
            )
            YY = metrics.pairwise.rbf_kernel(
                X_syn.numpy().reshape(len(X_syn), -1),
                X_syn.numpy().reshape(len(X_syn), -1),
                gamma,
            )
            XY = metrics.pairwise.rbf_kernel(
                X_gt.numpy().reshape(len(X_gt), -1),
                X_syn.numpy().reshape(len(X_syn), -1),
                gamma,
            )
            score = XX.mean() + YY.mean() - 2 * XY.mean()
        elif self.kernel == "polynomial":
            """
            MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
            """
            degree = 2
            gamma = 1
            coef0 = 0
            XX = metrics.pairwise.polynomial_kernel(
                X_gt.numpy().reshape(len(X_gt), -1),
                X_gt.numpy().reshape(len(X_gt), -1),
                degree,
                gamma,
                coef0,
            )
            YY = metrics.pairwise.polynomial_kernel(
                X_syn.numpy().reshape(len(X_syn), -1),
                X_syn.numpy().reshape(len(X_syn), -1),
                degree,
                gamma,
                coef0,
            )
            XY = metrics.pairwise.polynomial_kernel(
                X_gt.numpy().reshape(len(X_gt), -1),
                X_syn.numpy().reshape(len(X_syn), -1),
                degree,
                gamma,
                coef0,
            )
            score = XX.mean() + YY.mean() - 2 * XY.mean()
        else:
            raise ValueError(f"Unsupported kernel {self.kernel}")

        return {"joint": float(score)}


class JensenShannonDistance(StatisticalEvaluator):
    """Evaluate the average Jensen-Shannon distance (metric) between two probability arrays."""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(self, normalize: bool = True, **kwargs: Any) -> None:
        super().__init__(default_metric="marginal", **kwargs)

        self.normalize = normalize

    @staticmethod
    def name() -> str:
        return "jensenshannon_dist"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate_stats(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Tuple[Dict, Dict, Dict]:

        stats_gt = {}
        stats_syn = {}
        stats_ = {}

        for col in X_gt.columns:
            local_bins = min(self._n_histogram_bins, len(X_gt[col].unique()))
            X_gt_bin, gt_bins = pd.cut(X_gt[col], bins=local_bins, retbins=True)
            X_syn_bin = pd.cut(X_syn[col], bins=gt_bins)
            stats_gt[col], stats_syn[col] = X_gt_bin.value_counts(
                dropna=False, normalize=self.normalize
            ).align(
                X_syn_bin.value_counts(dropna=False, normalize=self.normalize),
                join="outer",
                axis=0,
                fill_value=0,
            )
            stats_gt[col] += 1
            stats_syn[col] += 1

            stats_[col] = jensenshannon(stats_gt[col], stats_syn[col])
            if np.isnan(stats_[col]):
                raise RuntimeError("NaNs in prediction")

        return stats_, stats_gt, stats_syn

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        stats_, _, _ = self._evaluate_stats(X_gt, X_syn)

        return {"marginal": sum(stats_.values()) / len(stats_.keys())}


class WassersteinDistance(StatisticalEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.WassersteinDistance
        :parts: 1

    Compare Wasserstein distance between original data and synthetic data.

    Args:
        X: original data
        X_syn: synthetically generated data

    Returns:
        WD_value: Wasserstein distance
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="joint", **kwargs)

    @staticmethod
    def name() -> str:
        return "wasserstein_dist"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        X_ = X.numpy().reshape(len(X), -1)
        X_syn_ = X_syn.numpy().reshape(len(X_syn), -1)

        if len(X_) > len(X_syn_):
            X_syn_ = np.concatenate(
                [X_syn_, np.zeros((len(X_) - len(X_syn_), X_.shape[1]))]
            )

        scaler = MinMaxScaler().fit(X_)

        X_ = scaler.transform(X_)
        X_syn_ = scaler.transform(X_syn_)

        X_ten = torch.from_numpy(X_)
        Xsyn_ten = torch.from_numpy(X_syn_)
        OT_solver = SamplesLoss(loss="sinkhorn")

        return {"joint": OT_solver(X_ten, Xsyn_ten).cpu().numpy().item()}


class PRDCScore(StatisticalEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.PRDCScore
        :parts: 1


    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        nearest_k: int.
    """

    def __init__(self, nearest_k: int = 5, **kwargs: Any) -> None:
        super().__init__(default_metric="precision", **kwargs)

        self.nearest_k = nearest_k

    @staticmethod
    def name() -> str:
        return "prdc"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        X_ = X.numpy().reshape(len(X), -1)
        X_syn_ = X_syn.numpy().reshape(len(X_syn), -1)

        # Default representation
        results = self._compute_prdc(X_, X_syn_)

        return results

    def _compute_pairwise_distance(
        self, data_x: np.ndarray, data_y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Args:
            data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
            data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
        Returns:
            numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
        """
        if data_y is None:
            data_y = data_x

        dists = metrics.pairwise_distances(data_x, data_y)
        return dists

    def _get_kth_value(
        self, unsorted: np.ndarray, k: int, axis: int = -1
    ) -> np.ndarray:
        """
        Args:
            unsorted: numpy.ndarray of any dimensionality.
            k: int
        Returns:
            kth values along the designated axis.
        """
        indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
        k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
        kth_values = k_smallests.max(axis=axis)
        return kth_values

    def _compute_nearest_neighbour_distances(
        self, input_features: np.ndarray, nearest_k: int
    ) -> np.ndarray:
        """
        Args:
            input_features: numpy.ndarray
            nearest_k: int
        Returns:
            Distances to kth nearest neighbours.
        """
        distances = self._compute_pairwise_distance(input_features)
        radii = self._get_kth_value(distances, k=nearest_k + 1, axis=-1)
        return radii

    def _compute_prdc(
        self, real_features: np.ndarray, fake_features: np.ndarray
    ) -> Dict:
        """
        Computes precision, recall, density, and coverage given two manifolds.
        Args:
            real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        Returns:
            dict of precision, recall, density, and coverage.
        """

        real_nearest_neighbour_distances = self._compute_nearest_neighbour_distances(
            real_features, self.nearest_k
        )
        fake_nearest_neighbour_distances = self._compute_nearest_neighbour_distances(
            fake_features, self.nearest_k
        )
        distance_real_fake = self._compute_pairwise_distance(
            real_features, fake_features
        )

        precision = (
            (
                distance_real_fake
                < np.expand_dims(real_nearest_neighbour_distances, axis=1)
            )
            .any(axis=0)
            .mean()
        )

        recall = (
            (
                distance_real_fake
                < np.expand_dims(fake_nearest_neighbour_distances, axis=0)
            )
            .any(axis=1)
            .mean()
        )

        density = (1.0 / float(self.nearest_k)) * (
            distance_real_fake
            < np.expand_dims(real_nearest_neighbour_distances, axis=1)
        ).sum(axis=0).mean()

        coverage = (
            distance_real_fake.min(axis=1) < real_nearest_neighbour_distances
        ).mean()

        return dict(
            precision=precision, recall=recall, density=density, coverage=coverage
        )


class AlphaPrecision(StatisticalEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.AlphaPrecision
        :parts: 1

    Evaluates the alpha-precision, beta-recall, and authenticity scores.

    The class evaluates the synthetic data using a tuple of three metrics:
    alpha-precision, beta-recall, and authenticity.
    Note that these metrics can be evaluated for each synthetic data point (which are useful for auditing and
    post-processing). Here we average the scores to reflect the overall quality of the data.
    The formal definitions can be found in the reference below:

    Alaa, Ahmed, Boris Van Breugel, Evgeny S. Saveliev, and Mihaela van der Schaar. "How faithful is your synthetic
    data? sample-level metrics for evaluating and auditing generative models."
    In International Conference on Machine Learning, pp. 290-306. PMLR, 2022.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="authenticity_OC", **kwargs)

    @staticmethod
    def name() -> str:
        return "alpha_precision"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def metrics(
        self,
        X: np.ndarray,
        X_syn: np.ndarray,
        emb_center: Optional[np.ndarray] = None,
    ) -> Tuple:
        if len(X) != len(X_syn):
            raise RuntimeError("The real and synthetic data must have the same length")

        if emb_center is None:
            emb_center = np.mean(X, axis=0)

        n_steps = 30
        alphas = np.linspace(0, 1, n_steps)

        Radii = np.quantile(np.sqrt(np.sum((X - emb_center) ** 2, axis=1)), alphas)

        synth_center = np.mean(X_syn, axis=0)

        alpha_precision_curve = []
        beta_coverage_curve = []

        synth_to_center = np.sqrt(np.sum((X_syn - emb_center) ** 2, axis=1))

        nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(X)
        real_to_real, _ = nbrs_real.kneighbors(X)

        nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(X_syn)
        real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(X)

        # Let us find closest real point to any real point, excluding itself (therefore 1 instead of 0)
        real_to_real = real_to_real[:, 1].squeeze()
        real_to_synth = real_to_synth.squeeze()
        real_to_synth_args = real_to_synth_args.squeeze()

        real_synth_closest = X_syn[real_to_synth_args]

        real_synth_closest_d = np.sqrt(
            np.sum((real_synth_closest - synth_center) ** 2, axis=1)
        )
        closest_synth_Radii = np.quantile(real_synth_closest_d, alphas)

        for k in range(len(Radii)):
            precision_audit_mask = synth_to_center <= Radii[k]
            alpha_precision = np.mean(precision_audit_mask)

            beta_coverage = np.mean(
                (
                    (real_to_synth <= real_to_real)
                    * (real_synth_closest_d <= closest_synth_Radii[k])
                )
            )

            alpha_precision_curve.append(alpha_precision)
            beta_coverage_curve.append(beta_coverage)

        # See which one is bigger

        authen = real_to_real[real_to_synth_args] < real_to_synth
        authenticity = np.mean(authen)

        Delta_precision_alpha = 1 - np.sum(
            np.abs(np.array(alphas) - np.array(alpha_precision_curve))
        ) / np.sum(alphas)

        if Delta_precision_alpha < 0:
            raise RuntimeError("negative value detected for Delta_precision_alpha")

        Delta_coverage_beta = 1 - np.sum(
            np.abs(np.array(alphas) - np.array(beta_coverage_curve))
        ) / np.sum(alphas)

        if Delta_coverage_beta < 0:
            raise RuntimeError("negative value detected for Delta_coverage_beta")

        return (
            alphas,
            alpha_precision_curve,
            beta_coverage_curve,
            Delta_precision_alpha,
            Delta_coverage_beta,
            authenticity,
        )

    def _normalize_covariates(
        self,
        X: DataLoader,
        X_syn: DataLoader,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """_normalize_covariates
        This is an internal method to replicate the old, naive method for evaluating
        AlphaPrecision.

        Args:
            X (DataLoader): The ground truth dataset.
            X_syn (DataLoader): The synthetic dataset.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: normalised version of the datasets
        """
        X_gt_norm = X.dataframe().copy()
        X_syn_norm = X_syn.dataframe().copy()
        if self._task_type != "survival_analysis":
            if hasattr(X, "target_column"):
                X_gt_norm = X_gt_norm.drop(columns=[X.target_column])
            if hasattr(X_syn, "target_column"):
                X_syn_norm = X_syn_norm.drop(columns=[X_syn.target_column])
        scaler = MinMaxScaler().fit(X_gt_norm)
        if hasattr(X, "target_column"):
            X_gt_norm_df = pd.DataFrame(
                scaler.transform(X_gt_norm),
                columns=[
                    col
                    for col in X.train().dataframe().columns
                    if col != X.target_column
                ],
            )
        else:
            X_gt_norm_df = pd.DataFrame(
                scaler.transform(X_gt_norm), columns=X.train().dataframe().columns
            )

        if hasattr(X_syn, "target_column"):
            X_syn_norm_df = pd.DataFrame(
                scaler.transform(X_syn_norm),
                columns=[
                    col
                    for col in X_syn.dataframe().columns
                    if col != X_syn.target_column
                ],
            )
        else:
            X_syn_norm_df = pd.DataFrame(
                scaler.transform(X_syn_norm), columns=X_syn.dataframe().columns
            )

        return (X_gt_norm_df, X_syn_norm_df)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:

        results = {}

        X_ = X.numpy().reshape(len(X), -1)
        X_syn_ = X_syn.numpy().reshape(len(X_syn), -1)

        # OneClass representation
        emb = "_OC"
        oneclass_model = self._get_oneclass_model(X_)
        X_ = self._oneclass_predict(oneclass_model, X_)
        X_syn_ = self._oneclass_predict(oneclass_model, X_syn_)
        emb_center = oneclass_model.c.detach().cpu().numpy()

        (
            alphas,
            alpha_precision_curve,
            beta_coverage_curve,
            Delta_precision_alpha,
            Delta_coverage_beta,
            authenticity,
        ) = self.metrics(X_, X_syn_, emb_center=emb_center)

        results[f"delta_precision_alpha{emb}"] = Delta_precision_alpha
        results[f"delta_coverage_beta{emb}"] = Delta_coverage_beta
        results[f"authenticity{emb}"] = authenticity

        X_df, X_syn_df = self._normalize_covariates(X, X_syn)
        (
            alphas_naive,
            alpha_precision_curve_naive,
            beta_coverage_curve_naive,
            Delta_precision_alpha_naive,
            Delta_coverage_beta_naive,
            authenticity_naive,
        ) = self.metrics(X_df.to_numpy(), X_syn_df.to_numpy(), emb_center=None)

        results["delta_precision_alpha_naive"] = Delta_precision_alpha_naive
        results["delta_coverage_beta_naive"] = Delta_coverage_beta_naive
        results["authenticity_naive"] = authenticity_naive

        return results


class SurvivalKMDistance(StatisticalEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.SurvivalKMDistance
        :parts: 1

    The distance between two Kaplan-Meier plots. Used for survival analysis"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="optimism", **kwargs)

    @staticmethod
    def name() -> str:
        return "survival_km_distance"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        if self._task_type != "survival_analysis":
            raise RuntimeError(
                f"The metric is valid only for survival analysis tasks, but got {self._task_type}"
            )
        if X.type() != "survival_analysis" or X_syn.type() != "survival_analysis":
            raise RuntimeError(
                f"The metric is valid only for survival analysis tasks, but got datasets {X.type()} and {X_syn.type()}"
            )

        _, real_T, real_E = X.unpack()
        _, syn_T, syn_E = X_syn.unpack()

        optimism, abs_optimism, sightedness = nonparametric_distance(
            (real_T, real_E), (syn_T, syn_E)
        )

        return {
            "optimism": optimism,
            "abs_optimism": abs_optimism,
            "sightedness": sightedness,
        }


class FrechetInceptionDistance(StatisticalEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.FrechetInceptionDistance
        :parts: 1

    Calculates the Frechet Inception Distance (FID) to evalulate GANs.

    Paper: GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.

    The FID metric calculates the distance between two distributions of images.
    Typically, we have summary statistics (mean & covariance matrix) of one of these distributions, while the 2nd distribution is given by a GAN.

    Adapted by Boris van Breugel(bv292@cam.ac.uk)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "fid"

    @staticmethod
    def direction() -> str:
        return "minimize"

    def _fit_gaussian(self, act: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculation of the statistics used by the FID.
        Params:
        -- act   : activations
        Returns:
        -- mu    : The mean over samples of the activations
        -- sigma : The covariance matrix of the activations
        """
        mu = np.mean(act, axis=0)
        sigma = np.cov(act.T)
        return mu, sigma

    def _calculate_frechet_distance(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1 : Numpy array containing the activations of the pool_3 layer of the
                 inception net ( like returned by the function 'get_predictions')
                 for generated samples.
        -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
                   on an representive data set.
        -- sigma1: The covariance matrix over activations of the pool_3 layer for
                   generated samples.
        -- sigma2: The covariance matrix over activations of the pool_3 layer,
                   precalcualted on an representive data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        if mu1.shape != mu2.shape:
            raise RuntimeError("Training and test mean vectors have different lengths")

        if sigma1.shape != sigma2.shape:
            raise RuntimeError(
                "Training and test covariances have different dimensions"
            )

        diff = mu1 - mu2

        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=2e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        if X.type() != "images":
            raise RuntimeError(
                f"The metric is valid only for image tasks, but got datasets {X.type()} and {X_syn.type()}"
            )

        X1 = X.numpy().reshape(len(X), -1)
        X2 = X_syn.numpy().reshape(len(X_syn), -1)

        mu1, cov1 = self._fit_gaussian(X1)
        mu2, cov2 = self._fit_gaussian(X2)
        score = self._calculate_frechet_distance(mu1, cov1, mu2, cov2)

        return {
            "score": score,
        }
