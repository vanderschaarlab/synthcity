# stdlib
from typing import Optional, Type

# third party
import numpy as np
import pandas as pd
import pytest
from lifelines.datasets import load_rossi
from sklearn.datasets import load_diabetes, load_iris

# synthcity absolute
from synthcity.metrics.eval_performance import (
    PerformanceEvaluatorLinear,
    PerformanceEvaluatorMLP,
    PerformanceEvaluatorXGB,
)
from synthcity.plugins import Plugin, Plugins
from synthcity.plugins.core.dataloader import (
    GenericDataLoader,
    SurvivalAnalysisDataLoader,
    TimeSeriesDataLoader,
    TimeSeriesSurvivalDataLoader,
    create_from_info,
)
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
from synthcity.utils.datasets.time_series.pbc import PBCDataloader


@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        PerformanceEvaluatorLinear,
        PerformanceEvaluatorMLP,
        PerformanceEvaluatorXGB,
    ],
)
def test_evaluate_performance_classifier(
    test_plugin: Plugin, evaluator_t: Type
) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X, target_column="target")
    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(100)

    evaluator = evaluator_t()
    good_score = evaluator.evaluate(
        Xloader,
        X_gen,
    )

    assert "gt" in good_score
    assert "syn_id" in good_score
    assert "syn_ood" in good_score

    assert good_score["gt"] > 0
    assert good_score["syn_id"] > 0
    assert good_score["syn_ood"] > 0

    sz = 100
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    score = evaluator.evaluate(
        Xloader,
        GenericDataLoader(X_rnd),
    )

    assert "gt" in score
    assert "syn_id" in score
    assert "syn_ood" in score

    assert score["syn_id"] < good_score["syn_id"]
    assert score["syn_ood"] < good_score["syn_ood"]

    assert evaluator.type() == "performance"
    assert evaluator.direction() == "maximize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        PerformanceEvaluatorLinear,
        PerformanceEvaluatorMLP,
        PerformanceEvaluatorXGB,
    ],
)
def test_evaluate_performance_regression(
    test_plugin: Plugin, evaluator_t: Type
) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    Xloader = GenericDataLoader(X, target_column="target")

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(1000)

    evaluator = evaluator_t()
    good_score = evaluator.evaluate(
        Xloader,
        X_gen,
    )

    assert "gt" in good_score
    assert "syn_id" in good_score
    assert "syn_ood" in good_score

    sz = 1000
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    score = evaluator.evaluate(
        Xloader,
        GenericDataLoader(X_rnd),
    )

    assert "gt" in score
    assert "syn_id" in score
    assert "syn_ood" in score

    assert score["syn_id"] <= good_score["syn_id"]
    assert score["syn_ood"] <= good_score["syn_ood"]


@pytest.mark.slow
@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        PerformanceEvaluatorLinear,
        PerformanceEvaluatorMLP,
        PerformanceEvaluatorXGB,
    ],
)
def test_evaluate_performance_survival_analysis(
    test_plugin: Plugin, evaluator_t: Type
) -> None:
    X = load_rossi()
    T = X["week"]
    time_horizons = np.linspace(T.min(), T.max(), num=4)[1:-1].tolist()

    Xloader = SurvivalAnalysisDataLoader(
        X,
        target_column="arrest",
        time_to_event_column="week",
        time_horizons=time_horizons,
    )
    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(100)

    evaluator = evaluator_t(
        task_type="survival_analysis",
    )
    good_score = evaluator.evaluate(
        Xloader,
        X_gen,
    )

    assert "gt.c_index" in good_score
    assert "gt.brier_score" in good_score
    assert "syn_id.c_index" in good_score
    assert "syn_id.brier_score" in good_score
    assert "syn_ood.c_index" in good_score
    assert "syn_ood.brier_score" in good_score

    sz = 100
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    X_rnd["arrest"] = 1
    score = evaluator.evaluate(
        Xloader,
        create_from_info(X_rnd, Xloader.info()),
    )

    assert "gt.c_index" in score
    assert "gt.brier_score" in score
    assert "syn_id.c_index" in score
    assert "syn_id.brier_score" in score
    assert "syn_ood.c_index" in score
    assert "syn_ood.brier_score" in score

    assert score["syn_id.c_index"] < 1
    assert score["syn_id.brier_score"] < 1
    assert score["syn_ood.c_index"] < 1
    assert score["syn_ood.brier_score"] < 1
    assert good_score["gt.c_index"] < 1
    assert good_score["gt.brier_score"] < 1


@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        PerformanceEvaluatorLinear,
        PerformanceEvaluatorXGB,
    ],
)
@pytest.mark.parametrize("target", [None, "target", "sepal width (cm)"])
def test_evaluate_performance_custom_labels(
    test_plugin: Plugin, evaluator_t: Type, target: Optional[str]
) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X, target_column="target")

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(100)

    evaluator = evaluator_t()

    good_score = evaluator.evaluate(
        Xloader,
        X_gen,
    )

    assert "gt" in good_score
    assert "syn_id" in good_score
    assert "syn_ood" in good_score


@pytest.mark.slow
@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        PerformanceEvaluatorMLP,
    ],
)
def test_evaluate_performance_time_series(
    test_plugin: Plugin, evaluator_t: Type
) -> None:
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
    data_gen = test_plugin.generate(100)

    evaluator = evaluator_t(
        task_type="time_series",
    )
    good_score = evaluator.evaluate(
        data,
        data_gen,
    )

    assert "gt" in good_score
    assert "syn_id" in good_score
    assert "syn_ood" in good_score

    sz = 100
    X_rnd = pd.DataFrame(np.random.randn(sz, len(data.columns)), columns=data.columns)
    X_rnd["arrest"] = 1
    score = evaluator.evaluate(
        data,
        create_from_info(X_rnd, data.info()),
    )

    assert "gt" in score
    assert "syn_id" in score
    assert "syn_ood" in score

    assert score["syn_id"] < 1
    assert score["syn_ood"] < 1
    assert good_score["gt"] < 1
    assert good_score["syn_id"] > score["syn_id"]
    assert good_score["syn_ood"] > score["syn_ood"]


@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        PerformanceEvaluatorLinear,
        PerformanceEvaluatorMLP,
        PerformanceEvaluatorXGB,
    ],
)
def test_evaluate_performance_time_series_survival(
    test_plugin: Plugin, evaluator_t: Type
) -> None:
    static_data, temporal_data, temporal_horizons, outcome = PBCDataloader().load()

    T, E = outcome

    data = TimeSeriesSurvivalDataLoader(
        temporal_data=temporal_data,
        temporal_horizons=temporal_horizons,
        static_data=static_data,
        T=T,
        E=E,
    )

    test_plugin.fit(data)
    data_gen = test_plugin.generate(len(temporal_data))

    evaluator = evaluator_t(
        task_type="time_series_survival",
    )

    good_score = evaluator.evaluate(
        data,
        data_gen,
    )
    assert "gt.c_index" in good_score
    assert "gt.brier_score" in good_score
    assert "syn_id.c_index" in good_score
    assert "syn_id.brier_score" in good_score
    assert "syn_ood.c_index" in good_score
    assert "syn_ood.brier_score" in good_score
    print(evaluator_t, good_score)

    assert good_score["syn_id.c_index"] < 1
    assert good_score["syn_ood.c_index"] < 1
