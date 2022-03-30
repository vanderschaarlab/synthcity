# stdlib
from typing import Callable, Tuple

# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics.sanity import (
    CommonRowsProportion,
    DataMismatchScore,
    InlierProbability,
    NearestSyntheticNeighborDistance,
    OutlierProbability,
)
from synthcity.plugins import Plugin, Plugins


def _eval_plugin(cbk: Callable, X: pd.DataFrame, X_syn: pd.DataFrame) -> Tuple:
    syn_score = cbk(
        X,
        X_syn,
    )

    sz = len(X_syn)
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    rnd_score = cbk(
        X,
        X_rnd,
    )

    return syn_score, rnd_score


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_data_mismatch_score(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    evaluator = DataMismatchScore()

    score = evaluator.evaluate(
        X,
        X_gen,
    )

    assert score == 0

    X_fail = X.head(100)
    X["target"] = "a"

    score = evaluator.evaluate(
        X,
        X_fail,
    )

    assert score > 0
    assert score < 1

    assert evaluator.type() == "sanity"
    assert evaluator.name() == "data_mismatch_score"
    assert evaluator.direction() == "minimize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_common_rows(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    evaluator = CommonRowsProportion()
    syn_score, rnd_score = _eval_plugin(evaluator.evaluate, X, X_gen)

    assert syn_score > 0
    assert syn_score < 1
    assert rnd_score == 0

    assert evaluator.type() == "sanity"
    assert evaluator.name() == "common_rows_proportion"
    assert evaluator.direction() == "minimize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_avg_distance_nearest_synth_neighbor(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    evaluator = NearestSyntheticNeighborDistance()

    syn_score, rnd_score = _eval_plugin(evaluator.evaluate, X, X_gen)

    assert syn_score > 0
    assert syn_score < 1

    assert rnd_score > 0
    assert rnd_score < 1
    assert syn_score < rnd_score

    assert evaluator.type() == "sanity"
    assert evaluator.name() == "nearest_syn_neighbor_distance"
    assert evaluator.direction() == "minimize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_inlier_probability(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    evaluator = InlierProbability()
    syn_score, rnd_score = _eval_plugin(evaluator.evaluate, X, X_gen)

    assert 0 < syn_score < 1
    assert 0 < rnd_score < 1
    assert syn_score > rnd_score

    assert evaluator.type() == "sanity"
    assert evaluator.name() == "inlier_probability"
    assert evaluator.direction() == "maximize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_outlier_probability(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    evaluator = OutlierProbability()
    syn_score, rnd_score = _eval_plugin(evaluator.evaluate, X, X_gen)

    assert 0 < syn_score < 1
    assert 0 < rnd_score < 1
    assert syn_score < rnd_score

    assert evaluator.type() == "sanity"
    assert evaluator.name() == "outlier_probability"
    assert evaluator.direction() == "minimize"
