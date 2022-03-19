# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from scipy.special import kl_div
from scipy.stats import chisquare, ks_2samp

# synthcity absolute
from synthcity.metrics._utils import get_freq


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_inv_kl_divergence(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_synth: pd.DataFrame, y_synth: pd.Series
) -> float:
    """Returns the average normalized Kullback–Leibler Divergence based metric."""
    freqs = get_freq(X_gt, X_synth)
    res = []
    for col in X_gt.columns:
        gt_freq, synth_freq = freqs[col]
        res.append(1 / (1 + np.sum(kl_div(gt_freq, synth_freq))))

    return np.mean(res)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_kolmogorov_smirnov_test(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_synth: pd.DataFrame, y_synth: pd.Series
) -> float:
    """Performs the Kolmogorov-Smirnov test for goodness of fit.

    Returns 1 if the distributions are identical.
    Returns 0 if the distributions are totally different.
    """

    res = []
    for col in X_gt.columns:
        statistic, _ = ks_2samp(X_gt[col], X_synth[col])
        res.append(1 - statistic)

    return np.mean(res)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_chi_squared_test(
    X_gt: pd.DataFrame, y_gt: pd.Series, X_synth: pd.DataFrame, y_synth: pd.Series
) -> float:
    """Performs the one-way chi-square test.

    Returns:
        The p-value. A small value indicates that we can reject the null hypothesis and that the distributions are different.
    """

    res = []
    freqs = get_freq(X_gt, X_synth)

    for col in X_gt.columns:
        gt_freq, synth_freq = freqs[col]
        try:
            _, pvalue = chisquare(gt_freq, synth_freq)
        except BaseException:
            pvalue = 0

        res.append(pvalue)

    return np.mean(res)
