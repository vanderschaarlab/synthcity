# stdlib
from typing import Any, Dict, Tuple

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
    def evaluate(self, X_gt: pd.DataFrame, X_syn: pd.DataFrame) -> Dict:
        freqs = get_frequency(X_gt, X_syn, n_histogram_bins=self._n_histogram_bins)
        res = []
        for col in X_gt.columns:
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
    def evaluate(self, X_gt: pd.DataFrame, X_syn: pd.DataFrame) -> Dict:
        res = []
        for col in X_gt.columns:
            statistic, _ = ks_2samp(X_gt[col], X_syn[col])
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
    def evaluate(self, X_gt: pd.DataFrame, X_syn: pd.DataFrame) -> Dict:
        res = []
        freqs = get_frequency(X_gt, X_syn, n_histogram_bins=self._n_histogram_bins)

        for col in X_gt.columns:
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
        X_gt: pd.DataFrame,
        X_syn: pd.DataFrame,
    ) -> Dict:
        if self.kernel == "linear":
            """
            MMD using linear kernel (i.e., k(x,y) = <x,y>)
            """
            delta_df = X_gt.mean(axis=0) - X_syn.mean(axis=0)
            delta = delta_df.values

            score = delta.dot(delta.T)
        elif self.kernel == "rbf":
            """
            MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
            """
            gamma = 1.0
            XX = metrics.pairwise.rbf_kernel(X_gt, X_gt, gamma)
            YY = metrics.pairwise.rbf_kernel(X_syn, X_syn, gamma)
            XY = metrics.pairwise.rbf_kernel(X_gt, X_syn, gamma)
            score = XX.mean() + YY.mean() - 2 * XY.mean()
        elif self.kernel == "polynomial":
            """
            MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
            """
            degree = 2
            gamma = 1
            coef0 = 0
            XX = metrics.pairwise.polynomial_kernel(X_gt, X_gt, degree, gamma, coef0)
            YY = metrics.pairwise.polynomial_kernel(X_syn, X_syn, degree, gamma, coef0)
            XY = metrics.pairwise.polynomial_kernel(X_gt, X_syn, degree, gamma, coef0)
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
        X_gt: pd.DataFrame,
        X_syn: pd.DataFrame,
    ) -> Dict:
        distances = []
        for col in X_syn.columns:
            if len(X_syn[col].unique()) < 15:
                continue
            syn_col = X_syn[col]
            gt_col = X_gt[col]

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
        X_gt: pd.DataFrame,
        X_syn: pd.DataFrame,
        normalizdde: bool = True,
    ) -> Dict:
        stats_, _, _ = self._evaluate_stats(X_gt, X_syn)

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
        X_gt: pd.DataFrame,
        X_syn: pd.DataFrame,
    ) -> Dict:
        stats_gt, stats_syn = self._evaluate_stats(X_gt, X_syn)

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
        X: pd.DataFrame,
        X_syn: pd.DataFrame,
    ) -> Dict:
        X_ten = torch.from_numpy(X.values)
        Xsyn_ten = torch.from_numpy(X_syn.values)
        OT_solver = SamplesLoss(loss="sinkhorn")

        return {"joint": OT_solver(X_ten, Xsyn_ten).cpu().numpy()}
