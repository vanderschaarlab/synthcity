# stdlib
from typing import Callable, Tuple

# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics.statistical import (
    evaluate_avg_jensenshannon_distance,
    evaluate_chi_squared_test,
    evaluate_feature_correlation,
    evaluate_inv_cdf_distance,
    evaluate_inv_kl_divergence,
    evaluate_kolmogorov_smirnov_test,
    evaluate_maximum_mean_discrepancy,
)
from synthcity.plugins import Plugin, Plugins


def _eval_plugin(cbk: Callable, X: pd.DataFrame, X_syn: pd.DataFrame) -> Tuple:
    syn_score = cbk(
        X.drop(columns=["target"]),
        X["target"],
        X_syn.drop(columns=["target"]),
        X_syn["target"],
    )

    sz = len(X_syn)
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    rnd_score = cbk(
        X.drop(columns=["target"]),
        X["target"],
        X_rnd.drop(columns=["target"]),
        X_rnd["target"],
    )

    return syn_score, rnd_score


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_kl_div(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(evaluate_inv_kl_divergence, X, X_gen)

    assert syn_score > 0
    assert rnd_score > 0
    assert syn_score > rnd_score


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_kolmogorov_smirnov_test(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(evaluate_kolmogorov_smirnov_test, X, X_gen)

    assert syn_score > 0
    assert rnd_score > 0
    assert syn_score > rnd_score


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_chi_squared_test(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(evaluate_chi_squared_test, X, X_gen)

    assert syn_score > 0
    assert rnd_score > 0
    assert syn_score > rnd_score


@pytest.mark.parametrize("kernel", ["linear", "rbf", "polynomial"])
@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_maximum_mean_discrepancy(kernel: str, test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(evaluate_maximum_mean_discrepancy, X, X_gen)

    assert syn_score > 0
    assert rnd_score > 0
    assert syn_score < rnd_score


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_inv_cdf_function(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(evaluate_inv_cdf_distance, X, X_gen)

    assert syn_score > 0
    assert rnd_score > 0
    assert syn_score < rnd_score


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_avg_jensenshannon_distance(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(evaluate_avg_jensenshannon_distance, X, X_gen)

    assert syn_score > 0
    assert rnd_score > 0
    assert syn_score < rnd_score


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_feature_correlation(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(evaluate_feature_correlation, X, X_gen)

    assert syn_score > 0
    assert rnd_score > 0
    assert syn_score < rnd_score
