# stdlib
from typing import Tuple, Type

# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics.eval_statistical import (
    AlphaPrecision,
    ChiSquaredTest,
    FeatureCorrelation,
    InverseCDFDistance,
    InverseKLDivergence,
    JensenShannonDistance,
    KolmogorovSmirnovTest,
    MaximumMeanDiscrepancy,
    PRDCScore,
    WassersteinDistance,
)
from synthcity.plugins import Plugin, Plugins


def _eval_plugin(evaluator_t: Type, X: pd.DataFrame, X_syn: pd.DataFrame) -> Tuple:
    evaluator = evaluator_t()

    syn_score = evaluator.evaluate(X, X, X_syn)

    sz = len(X_syn)
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    rnd_score = evaluator.evaluate(
        X,
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

    for key in syn_score:
        assert syn_score[key] > 0
        assert rnd_score[key] > 0
        assert syn_score[key] > rnd_score[key]

    assert InverseKLDivergence.name() == "inv_kl_divergence"
    assert InverseKLDivergence.type() == "stats"
    assert InverseKLDivergence.direction() == "maximize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_kolmogorov_smirnov_test(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(KolmogorovSmirnovTest, X, X_gen)

    for key in syn_score:
        assert syn_score[key] > 0
        assert rnd_score[key] > 0
        assert syn_score[key] > rnd_score[key]

    assert KolmogorovSmirnovTest.name() == "ks_test"
    assert KolmogorovSmirnovTest.type() == "stats"
    assert KolmogorovSmirnovTest.direction() == "maximize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_chi_squared_test(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(ChiSquaredTest, X, X_gen)

    for key in syn_score:
        assert syn_score[key] > 0
        assert rnd_score[key] > 0
        assert syn_score[key] > rnd_score[key]

    assert ChiSquaredTest.name() == "chi_squared_test"
    assert ChiSquaredTest.type() == "stats"
    assert ChiSquaredTest.direction() == "maximize"


@pytest.mark.parametrize("kernel", ["linear", "rbf", "polynomial"])
@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_maximum_mean_discrepancy(kernel: str, test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(MaximumMeanDiscrepancy, X, X_gen)

    for key in syn_score:
        assert syn_score[key] > 0
        assert rnd_score[key] > 0
        assert syn_score[key] < rnd_score[key]

    assert MaximumMeanDiscrepancy.name() == "max_mean_discrepancy"
    assert MaximumMeanDiscrepancy.type() == "stats"
    assert MaximumMeanDiscrepancy.direction() == "minimize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_inv_cdf_function(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(InverseCDFDistance, X, X_gen)

    for key in syn_score:
        assert syn_score[key] > 0
        assert rnd_score[key] > 0
        assert syn_score[key] < rnd_score[key]

    assert InverseCDFDistance.name() == "inv_cdf_dist"
    assert InverseCDFDistance.type() == "stats"
    assert InverseCDFDistance.direction() == "minimize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_avg_jensenshannon_distance(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(JensenShannonDistance, X, X_gen)

    for key in syn_score:
        assert syn_score[key] > 0
        assert rnd_score[key] > 0
        assert syn_score[key] < rnd_score[key]

    assert JensenShannonDistance.name() == "jensenshannon_dist"
    assert JensenShannonDistance.type() == "stats"
    assert JensenShannonDistance.direction() == "minimize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_feature_correlation(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(FeatureCorrelation, X, X_gen)

    for key in syn_score:
        assert syn_score[key] > 0
        assert rnd_score[key] > 0
        assert syn_score[key] < rnd_score[key]

    assert FeatureCorrelation.name() == "feature_corr"
    assert FeatureCorrelation.type() == "stats"
    assert FeatureCorrelation.direction() == "minimize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_wasserstein_distance(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(WassersteinDistance, X, X_gen)

    for key in syn_score:
        assert syn_score[key] > 0
        assert rnd_score[key] > 0
        assert syn_score[key] < rnd_score[key]

    assert WassersteinDistance.name() == "wasserstein_dist"
    assert WassersteinDistance.type() == "stats"
    assert WassersteinDistance.direction() == "minimize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_prdc(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(PRDCScore, X, X_gen)

    for key in syn_score:
        assert syn_score[key] >= 0
        assert rnd_score[key] >= 0
        assert syn_score[key] > rnd_score[key]

    assert PRDCScore.name() == "prdc"
    assert PRDCScore.type() == "stats"
    assert PRDCScore.direction() == "maximize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_alpha_precision(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(len(X))

    syn_score, rnd_score = _eval_plugin(AlphaPrecision, X, X_gen)
    print(syn_score, rnd_score)

    assert syn_score["delta_precision_alpha"] > rnd_score["delta_precision_alpha"]
    assert syn_score["authenticity"] < rnd_score["authenticity"]

    assert AlphaPrecision.name() == "alpha_precision"
    assert AlphaPrecision.type() == "stats"
    assert AlphaPrecision.direction() == "maximize"
