# stdlib
from typing import Any, Dict, Optional, Tuple

# third party
import numpy as np
import pandas as pd
import torch
from copulas.univariate.base import Univariate
from dython.nominal import associations
from geomloss import SamplesLoss
from pydantic import validate_arguments
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
from scipy.stats import chisquare, ks_2samp
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors

# synthcity absolute
from synthcity.metrics._utils import get_frequency
from synthcity.metrics.core import MetricEvaluator


class InverseKLDivergence(MetricEvaluator):
    """Returns the average inverse of the Kullback–Leibler Divergence metric.

    Score:
        0: the datasets are from different distributions.
        1: the datasets are from the same distribution.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        return "stats"

    @staticmethod
    def name() -> str:
        return "inv_kl_divergence"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self, X_gt_train: pd.DataFrame, X_gt_test: pd.DataFrame, X_syn: pd.DataFrame
    ) -> Dict:
        freqs = get_frequency(
            X_gt_train, X_syn, n_histogram_bins=self._n_histogram_bins
        )
        res = []
        for col in X_gt_train.columns:
            gt_freq, synth_freq = freqs[col]
            res.append(1 / (1 + np.sum(kl_div(gt_freq, synth_freq))))

        return {"marginal": float(self.reduction()(res))}


class KolmogorovSmirnovTest(MetricEvaluator):
    """Performs the Kolmogorov-Smirnov test for goodness of fit.

    Score:
        0: the distributions are totally different.
        1: the distributions are identical.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        return "stats"

    @staticmethod
    def name() -> str:
        return "ks_test"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self, X_gt_train: pd.DataFrame, X_gt_test: pd.DataFrame, X_syn: pd.DataFrame
    ) -> Dict:
        res = []
        for col in X_gt_train.columns:
            statistic, _ = ks_2samp(X_gt_train[col], X_syn[col])
            res.append(1 - statistic)

        return {"marginal": float(self.reduction()(res))}


class ChiSquaredTest(MetricEvaluator):
    """Performs the one-way chi-square test.

    Returns:
        The p-value. A small value indicates that we can reject the null hypothesis and that the distributions are different.

    Score:
        0: the distributions are different
        1: the distributions are identical.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        return "stats"

    @staticmethod
    def name() -> str:
        return "chi_squared_test"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self, X_gt_train: pd.DataFrame, X_gt_test: pd.DataFrame, X_syn: pd.DataFrame
    ) -> Dict:
        res = []
        freqs = get_frequency(
            X_gt_train, X_syn, n_histogram_bins=self._n_histogram_bins
        )

        for col in X_gt_train.columns:
            gt_freq, synth_freq = freqs[col]
            try:
                _, pvalue = chisquare(gt_freq, synth_freq)
            except BaseException:
                pvalue = 0

            res.append(pvalue)

        return {"marginal": float(self.reduction()(res))}


class MaximumMeanDiscrepancy(MetricEvaluator):
    """Empirical maximum mean discrepancy. The lower the result the more evidence that distributions are the same.

    Args:
        kernel: "rbf", "linear" or "polynomial"

    Score:
        0: The distributions are the same.
        1: The distributions are totally different.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(self, kernel: str = "rbf", **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.kernel = kernel

    @staticmethod
    def type() -> str:
        return "stats"

    @staticmethod
    def name() -> str:
        return "max_mean_discrepancy"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt_train: pd.DataFrame,
        X_gt_test: pd.DataFrame,
        X_syn: pd.DataFrame,
    ) -> Dict:
        if self.kernel == "linear":
            """
            MMD using linear kernel (i.e., k(x,y) = <x,y>)
            """
            delta_df = X_gt_train.mean(axis=0) - X_syn.mean(axis=0)
            delta = delta_df.values

            score = delta.dot(delta.T)
        elif self.kernel == "rbf":
            """
            MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
            """
            gamma = 1.0
            XX = metrics.pairwise.rbf_kernel(X_gt_train, X_gt_train, gamma)
            YY = metrics.pairwise.rbf_kernel(X_syn, X_syn, gamma)
            XY = metrics.pairwise.rbf_kernel(X_gt_train, X_syn, gamma)
            score = XX.mean() + YY.mean() - 2 * XY.mean()
        elif self.kernel == "polynomial":
            """
            MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
            """
            degree = 2
            gamma = 1
            coef0 = 0
            XX = metrics.pairwise.polynomial_kernel(
                X_gt_train, X_gt_train, degree, gamma, coef0
            )
            YY = metrics.pairwise.polynomial_kernel(X_syn, X_syn, degree, gamma, coef0)
            XY = metrics.pairwise.polynomial_kernel(
                X_gt_train, X_syn, degree, gamma, coef0
            )
            score = XX.mean() + YY.mean() - 2 * XY.mean()
        else:
            raise ValueError(f"Unsupported kernel {self.kernel}")

        return {"joint": float(score)}


class InverseCDFDistance(MetricEvaluator):
    """Evaluate the distance between continuous features."""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(self, p: int = 2, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.p = p

    @staticmethod
    def type() -> str:
        return "stats"

    @staticmethod
    def name() -> str:
        return "inv_cdf_dist"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt_train: pd.DataFrame,
        X_gt_test: pd.DataFrame,
        X_syn: pd.DataFrame,
    ) -> Dict:
        distances = []
        for col in X_syn.columns:
            if len(X_syn[col].unique()) < 15:
                continue
            syn_col = X_syn[col]
            gt_col = X_gt_train[col]

            predictor = Univariate()
            predictor.fit(syn_col)

            syn_percentiles = predictor.cdf(np.array(syn_col))
            gt_percentiles = predictor.cdf(np.array(gt_col))
            distances.append(
                np.mean(abs(syn_percentiles - gt_percentiles[1]) ** self.p)
            )

        return {"marginal": float(self.reduction()(distances))}


class JensenShannonDistance(MetricEvaluator):
    """Evaluate the average Jensen-Shannon distance (metric) between two probability arrays."""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(self, normalize: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.normalize = normalize

    @staticmethod
    def type() -> str:
        return "stats"

    @staticmethod
    def name() -> str:
        return "jensenshannon_dist"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate_stats(
        self,
        X_gt: pd.DataFrame,
        X_syn: pd.DataFrame,
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
            stats_[col] = jensenshannon(stats_gt[col], stats_syn[col])

        return stats_, stats_gt, stats_syn

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt_train: pd.DataFrame,
        X_gt_test: pd.DataFrame,
        X_syn: pd.DataFrame,
        normalizdde: bool = True,
    ) -> Dict:
        stats_, _, _ = self._evaluate_stats(X_gt_train, X_syn)

        return {"marginal": sum(stats_.values()) / len(stats_.keys())}


class FeatureCorrelation(MetricEvaluator):
    """Evaluate the correlation/strength-of-association of features in data-set with both categorical and continuous features using: * Pearson's R for continuous-continuous cases ** Cramer's V or Theil's U for categorical-categorical cases."""

    def __init__(
        self, nom_nom_assoc: str = "theil", nominal_columns: str = "auto", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.nom_nom_assoc = nom_nom_assoc
        self.nominal_columns = nominal_columns

    @staticmethod
    def type() -> str:
        return "stats"

    @staticmethod
    def name() -> str:
        return "feature_corr"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate_stats(
        self,
        X_gt: pd.DataFrame,
        X_syn: pd.DataFrame,
    ) -> Tuple[Dict, Dict]:
        stats_gt = associations(
            X_gt,
            nom_nom_assoc=self.nom_nom_assoc,
            nominal_columns=self.nominal_columns,
            nan_replace_value="nan",
            compute_only=True,
        )["corr"]
        stats_syn = associations(
            X_syn,
            nom_nom_assoc=self.nom_nom_assoc,
            nominal_columns=self.nominal_columns,
            nan_replace_value="nan",
            compute_only=True,
        )["corr"]

        return stats_gt, stats_syn

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt_train: pd.DataFrame,
        X_gt_test: pd.DataFrame,
        X_syn: pd.DataFrame,
    ) -> Dict:
        stats_gt, stats_syn = self._evaluate_stats(X_gt_train, X_syn)

        return {"joint": np.linalg.norm(stats_gt - stats_syn, "fro")}


class WassersteinDistance(MetricEvaluator):
    """Compare Wasserstein distance between original data and synthetic data.

    Args:
        X: original data
        X_syn: synthetically generated data

    Returns:
        WD_value: Wasserstein distance
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        return "stats"

    @staticmethod
    def name() -> str:
        return "wasserstein_dist"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        X_syn: pd.DataFrame,
    ) -> Dict:
        X_ten = torch.from_numpy(X_train.values)
        Xsyn_ten = torch.from_numpy(X_syn.values.astype(float))
        OT_solver = SamplesLoss(loss="sinkhorn")

        return {"joint": OT_solver(X_ten, Xsyn_ten).cpu().numpy()}


class PRDCScore(MetricEvaluator):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        nearest_k: int.
    """

    def __init__(self, nearest_k: int = 5, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.nearest_k = nearest_k

    @staticmethod
    def type() -> str:
        return "stats"

    @staticmethod
    def name() -> str:
        return "prdc"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        X_syn: pd.DataFrame,
    ) -> Dict:
        return self._compute_prdc(np.asarray(X_train), np.asarray(X_syn))

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
        dists = metrics.pairwise_distances(data_x, data_y, metric="euclidean")
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


class AlphaPrecision(MetricEvaluator):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        return "stats"

    @staticmethod
    def name() -> str:
        return "alpha_precision"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def metrics(
        self,
        X_df: pd.DataFrame,
        X_syn_df: pd.DataFrame,
    ) -> Tuple:
        X = X_df.values.astype(float)
        X_syn = X_syn_df.values.astype(float)
        assert len(X) == len(
            X_syn
        ), "The real and synthetic data mush have the same length"

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

        Delta_precision_alpha = 1 - 2 * np.sum(
            np.abs(np.array(alphas) - np.array(alpha_precision_curve))
        ) * (alphas[1] - alphas[0])
        Delta_coverage_beta = 1 - 2 * np.sum(
            np.abs(np.array(alphas) - np.array(beta_coverage_curve))
        ) * (alphas[1] - alphas[0])

        return (
            alphas,
            alpha_precision_curve,
            beta_coverage_curve,
            Delta_precision_alpha,
            Delta_coverage_beta,
            authenticity,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_train_df: pd.DataFrame,
        X_test_df: pd.DataFrame,
        X_syn_df: pd.DataFrame,
    ) -> Dict:

        (
            alphas,
            alpha_precision_curve,
            beta_coverage_curve,
            Delta_precision_alpha,
            Delta_coverage_beta,
            authenticity,
        ) = self.metrics(X_train_df, X_syn_df)

        return {
            "delta_precision_alpha": Delta_precision_alpha,
            "delta_coverage_beta": Delta_coverage_beta,
            "authenticity": authenticity,
        }
