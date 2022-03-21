# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from scipy.special import kl_div
from scipy.stats import chisquare, ks_2samp
from sklearn import metrics

# synthcity absolute
from synthcity.metrics._utils import get_freq


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_inv_kl_divergence(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_syn: pd.DataFrame, y_synth: pd.Series
) -> float:
    """Returns the average inverse of the Kullback–Leibler Divergence metric.

    Score:
        0: the datasets are from different distributions.
        1: the datasets are from the same distribution.
    """
    freqs = get_freq(X_gt, X_syn)
    res = []
    for col in X_gt.columns:
        gt_freq, synth_freq = freqs[col]
        res.append(1 / (1 + np.sum(kl_div(gt_freq, synth_freq))))

    return np.mean(res)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_kolmogorov_smirnov_test(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_syn: pd.DataFrame, y_synth: pd.Series
) -> float:
    """Performs the Kolmogorov-Smirnov test for goodness of fit.

    Score:
        0: the distributions are totally different.
        1: the distributions are identical.
    """

    res = []
    for col in X_gt.columns:
        statistic, _ = ks_2samp(X_gt[col], X_syn[col])
        res.append(1 - statistic)

    return np.mean(res)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_chi_squared_test(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_syn: pd.DataFrame, y_synth: pd.Series
) -> float:
    """Performs the one-way chi-square test.

    Returns:
        The p-value. A small value indicates that we can reject the null hypothesis and that the distributions are different.

    Score:
        0: the distributions are different
        1: the distributions are identical.
    """

    res = []
    freqs = get_freq(X_gt, X_syn)

    for col in X_gt.columns:
        gt_freq, synth_freq = freqs[col]
        try:
            _, pvalue = chisquare(gt_freq, synth_freq)
        except BaseException:
            pvalue = 0

        res.append(pvalue)

    return np.mean(res)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_maximum_mean_discrepancy(
    X_gt: pd.DataFrame,
    y_gt: pd.Series,
    X_syn: pd.DataFrame,
    y_synth: pd.Series,
    kernel: str = "rbf",
) -> float:
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    X_gt["target"] = y_gt
    X_syn["target"] = y_synth

    if kernel == "linear":
        """
        MMD using linear kernel (i.e., k(x,y) = <x,y>)
        """
        delta_df = X_gt.mean(axis=0) - X_syn.mean(axis=0)
        delta = delta_df.values

        return delta.dot(delta.T)
    elif kernel == "rbf":
        """
        MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
        """
        gamma = 1.0
        XX = metrics.pairwise.rbf_kernel(X_gt, X_gt, gamma)
        YY = metrics.pairwise.rbf_kernel(X_syn, X_syn, gamma)
        XY = metrics.pairwise.rbf_kernel(X_gt, X_syn, gamma)
        return XX.mean() + YY.mean() - 2 * XY.mean()
    elif kernel == "polynomial":
        """
        MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
        """
        degree = 2
        gamma = 1
        coef0 = 0
        XX = metrics.pairwise.polynomial_kernel(X_gt, X_gt, degree, gamma, coef0)
        YY = metrics.pairwise.polynomial_kernel(X_syn, X_syn, degree, gamma, coef0)
        XY = metrics.pairwise.polynomial_kernel(X_gt, X_syn, degree, gamma, coef0)
        return XX.mean() + YY.mean() - 2 * XY.mean()
    else:
        raise ValueError(f"Unsupported kernel {kernel}")
