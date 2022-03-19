# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics.statistical import (
    evaluate_chi_squared_test,
    evaluate_inv_kl_divergence,
    evaluate_kolmogorov_smirnov_test,
)
from synthcity.plugins import Plugin, Plugins


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_kl_div(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    bad_score = evaluate_inv_kl_divergence(
        X.drop(columns=["target"]),
        X["target"],
        X_gen.drop(columns=["target"]),
        X_gen["target"],
    )

    assert bad_score > 0

    sz = 100
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    score = evaluate_inv_kl_divergence(
        X.drop(columns=["target"]),
        X["target"],
        X_rnd.drop(columns=["target"]),
        X_rnd["target"],
    )

    assert score > 0
    assert bad_score > score


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_kolmogorov_smirnov_test(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    bad_score = evaluate_kolmogorov_smirnov_test(
        X.drop(columns=["target"]),
        X["target"],
        X_gen.drop(columns=["target"]),
        X_gen["target"],
    )

    assert bad_score > 0

    sz = 100
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    score = evaluate_kolmogorov_smirnov_test(
        X.drop(columns=["target"]),
        X["target"],
        X_rnd.drop(columns=["target"]),
        X_rnd["target"],
    )

    assert score > 0
    assert bad_score > score


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_chi_squared_test(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    bad_score = evaluate_chi_squared_test(
        X.drop(columns=["target"]),
        X["target"],
        X_gen.drop(columns=["target"]),
        X_gen["target"],
    )

    assert bad_score > 0

    sz = 1000
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    score = evaluate_chi_squared_test(
        X.drop(columns=["target"]),
        X["target"],
        X_rnd.drop(columns=["target"]),
        X_rnd["target"],
    )

    assert score > 0
    assert bad_score > score
