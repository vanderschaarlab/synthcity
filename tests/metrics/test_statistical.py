# stdlib
from typing import Tuple, Type

# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics.statistical import (
    ChiSquaredTest,
    FeatureCorrelation,
    InverseCDFDistance,
    InverseKLDivergence,
    JensenShannonDistance,
    KolmogorovSmirnovTest,
    MaximumMeanDiscrepancy,
    WassersteinDistance,
)
from synthcity.plugins import Plugin, Plugins


def _eval_plugin(evaluator_t: Type, X: pd.DataFrame, X_syn: pd.DataFrame) -> Tuple:
    evaluator = evaluator_t()

    syn_score = evaluator.evaluate(X, X_syn)

    sz = len(X_syn)
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    rnd_score = evaluator.evaluate(
        X,
        X_rnd,
    )

    return syn_score, rnd_score


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_kl_div(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(InverseKLDivergence, X, X_gen)

    assert syn_score > 0
    assert rnd_score > 0
    assert syn_score > rnd_score

    assert InverseKLDivergence.name() == "inverse_kl_divergence"
    assert InverseKLDivergence.type() == "statistical.marginal"
    assert InverseKLDivergence.direction() == "maximize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_kolmogorov_smirnov_test(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(KolmogorovSmirnovTest, X, X_gen)

    assert syn_score > 0
    assert rnd_score > 0
    assert syn_score > rnd_score

    assert KolmogorovSmirnovTest.name() == "kolmogorov_smirnov_test"
    assert KolmogorovSmirnovTest.type() == "statistical.marginal"
    assert KolmogorovSmirnovTest.direction() == "maximize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_chi_squared_test(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(ChiSquaredTest, X, X_gen)

    assert syn_score > 0
    assert rnd_score > 0
    assert syn_score > rnd_score

    assert ChiSquaredTest.name() == "chi_squared_test"
    assert ChiSquaredTest.type() == "statistical.marginal"
    assert ChiSquaredTest.direction() == "maximize"


@pytest.mark.parametrize("kernel", ["linear", "rbf", "polynomial"])
@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_maximum_mean_discrepancy(kernel: str, test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(MaximumMeanDiscrepancy, X, X_gen)

    assert syn_score > 0
    assert rnd_score > 0
    assert syn_score < rnd_score

    assert MaximumMeanDiscrepancy.name() == "maximum_mean_discrepancy"
    assert MaximumMeanDiscrepancy.type() == "statistical.joint"
    assert MaximumMeanDiscrepancy.direction() == "minimize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_inv_cdf_function(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(InverseCDFDistance, X, X_gen)

    assert syn_score > 0
    assert rnd_score > 0
    assert syn_score < rnd_score

    assert InverseCDFDistance.name() == "inverse_cdf_distance"
    assert InverseCDFDistance.type() == "statistical.marginal"
    assert InverseCDFDistance.direction() == "minimize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_avg_jensenshannon_distance(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(JensenShannonDistance, X, X_gen)

    assert syn_score > 0
    assert rnd_score > 0
    assert syn_score < rnd_score

    assert JensenShannonDistance.name() == "jensenshannon_distance"
    assert JensenShannonDistance.type() == "statistical.marginal"
    assert JensenShannonDistance.direction() == "minimize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_feature_correlation(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(FeatureCorrelation, X, X_gen)

    assert syn_score > 0
    assert rnd_score > 0
    assert syn_score < rnd_score

    assert FeatureCorrelation.name() == "feature_correlation"
    assert FeatureCorrelation.type() == "statistical.joint"
    assert FeatureCorrelation.direction() == "minimize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_wasserstein_distance(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(WassersteinDistance, X, X_gen)

    assert syn_score > 0
    assert rnd_score > 0
    assert syn_score < rnd_score

    assert WassersteinDistance.name() == "wasserstein_distance"
    assert WassersteinDistance.type() == "statistical.joint"
    assert WassersteinDistance.direction() == "minimize"
