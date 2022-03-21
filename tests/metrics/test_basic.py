# stdlib
from typing import Callable, Tuple

# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics.basic import (
    evaluate_avg_distance_nearest_synth_neighbor,
    evaluate_common_rows_proportion,
    evaluate_data_mismatch_score,
    evaluate_inlier_probability,
    evaluate_outlier_probability,
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
def test_evaluate_data_mismatch_score(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    score = evaluate_data_mismatch_score(
        X.drop(columns=["target"]),
        X["target"],
        X_gen.drop(columns=["target"]),
        X_gen["target"],
    )

    assert score == 0

    X_fail = X.head(100)
    X["target"] = "a"

    score = evaluate_data_mismatch_score(
        X.drop(columns=["target"]),
        X["target"],
        X_fail.drop(columns=["target"]),
        X_fail["target"],
    )

    assert score > 0
    assert score < 1


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_common_rows(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    syn_score, rnd_score = _eval_plugin(evaluate_common_rows_proportion, X, X_gen)

    assert syn_score > 0
    assert syn_score < 1
    assert rnd_score == 0


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_avg_distance_nearest_synth_neighbor(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    syn_score, rnd_score = _eval_plugin(
        evaluate_avg_distance_nearest_synth_neighbor, X, X_gen
    )

    assert syn_score > 0
    assert syn_score < 1

    assert rnd_score > 0
    assert rnd_score < 1
    assert syn_score < rnd_score


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_inlier_probability(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    syn_score, rnd_score = _eval_plugin(evaluate_inlier_probability, X, X_gen)

    assert 0 < syn_score < 1
    assert 0 < rnd_score < 1
    assert syn_score > rnd_score


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_outlier_probability(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    syn_score, rnd_score = _eval_plugin(evaluate_outlier_probability, X, X_gen)

    assert 0 < syn_score < 1
    assert 0 < rnd_score < 1
    assert syn_score < rnd_score
