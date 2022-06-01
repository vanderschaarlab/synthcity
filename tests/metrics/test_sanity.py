# stdlib
from typing import Callable, Tuple

# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics.eval_sanity import (
    CloseValuesProbability,
    CommonRowsProportion,
    DataMismatchScore,
    DistantValuesProbability,
    NearestSyntheticNeighborDistance,
)
from synthcity.plugins import Plugin, Plugins
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader


def _eval_plugin(cbk: Callable, X: DataLoader, X_syn: DataLoader) -> Tuple:
    syn_score = cbk(
        X,
        X_syn,
    )

    sz = len(X_syn)
    X_rnd = GenericDataLoader(
        pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    )
    rnd_score = cbk(
        X,
        X_rnd,
    )

    return syn_score, rnd_score


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_data_mismatch_score(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X)

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(100)

    evaluator = DataMismatchScore()

    score = evaluator.evaluate(
        Xloader,
        X_gen,
    )

    for key in score:
        assert score[key] == 0

    X_fail = X.head(100)
    X["target"] = "a"

    score = evaluator.evaluate(
        GenericDataLoader(X),
        GenericDataLoader(X_fail),
    )

    for key in score:
        assert score[key] > 0
        assert score[key] < 1

    assert evaluator.type() == "sanity"
    assert evaluator.name() == "data_mismatch"
    assert evaluator.direction() == "minimize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_common_rows(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X)

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(100)

    evaluator = CommonRowsProportion()
    syn_score, rnd_score = _eval_plugin(evaluator.evaluate, Xloader, X_gen)

    for key in syn_score:
        assert syn_score[key] > 0
        assert syn_score[key] < 1
        assert rnd_score[key] == 0

    assert evaluator.type() == "sanity"
    assert evaluator.name() == "common_rows_proportion"
    assert evaluator.direction() == "minimize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_avg_distance_nearest_synth_neighbor(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X)

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(100)

    evaluator = NearestSyntheticNeighborDistance()

    syn_score, rnd_score = _eval_plugin(evaluator.evaluate, Xloader, X_gen)

    for key in syn_score:
        assert syn_score[key] > 0
        assert syn_score[key] < 1

        assert rnd_score[key] > 0
        assert rnd_score[key] < 1
        assert syn_score[key] < rnd_score[key]

    assert evaluator.type() == "sanity"
    assert evaluator.name() == "nearest_syn_neighbor_distance"
    assert evaluator.direction() == "minimize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_close_values(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X)

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(100)

    evaluator = CloseValuesProbability()
    syn_score, rnd_score = _eval_plugin(evaluator.evaluate, Xloader, X_gen)

    for key in syn_score:
        assert 0 < syn_score[key] < 1
        assert 0 < rnd_score[key] < 1
        assert syn_score[key] > rnd_score[key]

    assert evaluator.type() == "sanity"
    assert evaluator.name() == "close_values_probability"
    assert evaluator.direction() == "maximize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_distant_values(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X)

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(100)

    evaluator = DistantValuesProbability()
    syn_score, rnd_score = _eval_plugin(evaluator.evaluate, Xloader, X_gen)

    for key in syn_score:
        assert 0 < syn_score[key] < 1
        assert 0 < rnd_score[key] < 1
        assert syn_score[key] < rnd_score[key]

    assert evaluator.type() == "sanity"
    assert evaluator.name() == "distant_values_probability"
    assert evaluator.direction() == "minimize"
