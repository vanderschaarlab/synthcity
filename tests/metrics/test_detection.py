# stdlib
from typing import Type

# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics.eval_detection import (
    SyntheticDetectionGMM,
    SyntheticDetectionMLP,
    SyntheticDetectionXGB,
)
from synthcity.plugins import Plugin, Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader, TimeSeriesDataLoader
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader


@pytest.mark.parametrize("reduction", ["mean", "max", "min"])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        SyntheticDetectionXGB,
    ],
)
def test_detect_reduction(reduction: str, evaluator_t: Type) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X)

    test_plugin = Plugins().get("marginal_distributions")
    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(100)

    evaluator = evaluator_t(reduction=reduction)

    score = evaluator.evaluate(
        Xloader,
        X_gen,
    )

    assert reduction in score


@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        SyntheticDetectionXGB,
        SyntheticDetectionGMM,
        SyntheticDetectionMLP,
    ],
)
def test_detect_synth_generic(test_plugin: Plugin, evaluator_t: Type) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X)

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(100)

    evaluator = evaluator_t()

    good_score = evaluator.evaluate(
        Xloader,
        X_gen,
    )["mean"]

    assert good_score > 0
    assert good_score <= 1

    sz = 100
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    score = evaluator.evaluate(
        Xloader,
        GenericDataLoader(X_rnd),
    )["mean"]

    assert score > 0
    assert score <= 1
    assert good_score < score

    assert evaluator.type() == "detection"
    assert evaluator.direction() == "minimize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        SyntheticDetectionXGB,
    ],
)
def test_detect_synth_timeseries(test_plugin: Plugin, evaluator_t: Type) -> None:
    (
        static_data,
        temporal_data,
        temporal_horizons,
        outcome,
    ) = GoogleStocksDataloader().load()
    data = TimeSeriesDataLoader(
        temporal_data=temporal_data,
        temporal_horizons=temporal_horizons,
        static_data=static_data,
        outcome=outcome,
    )

    test_plugin.fit(data)
    data_gen = test_plugin.generate(200)

    evaluator = evaluator_t()

    good_score = evaluator.evaluate(
        data,
        data_gen,
    )["mean"]

    assert good_score > 0
    assert good_score <= 1

    sz = 200
    data_rnd = TimeSeriesDataLoader.from_info(
        pd.DataFrame(np.random.randn(sz, len(data.columns)), columns=data.columns),
        data.info(),
    )

    score = evaluator.evaluate(
        data,
        data_rnd,
    )["mean"]

    assert score > 0
    assert score <= 1
    assert good_score < score

    assert evaluator.type() == "detection"
    assert evaluator.direction() == "minimize"
