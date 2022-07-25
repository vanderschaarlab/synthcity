# stdlib
from pathlib import Path

# third party
import pytest
from lifelines.datasets import load_rossi
from sklearn.datasets import load_diabetes, load_iris

# synthcity absolute
from synthcity.benchmark import Benchmarks
from synthcity.plugins.core.dataloader import (
    GenericDataLoader,
    SurvivalAnalysisDataLoader,
)


def test_benchmark_sanity() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    scores = Benchmarks.evaluate(
        [
            ("test1", "marginal_distributions", {}),
            ("test2", "dummy_sampler", {}),
        ],
        GenericDataLoader(X, sensitive_columns=["sex"]),
        metrics={"sanity": ["common_rows_proportion", "data_mismatch_score"]},
    )

    Benchmarks.print(scores)


def test_benchmark_invalid_plugin() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    with pytest.raises(ValueError):
        Benchmarks.evaluate(
            [
                ("test1", "invalid", {}),
                ("test2", "dummy_sampler", {}),
            ],
            GenericDataLoader(X, sensitive_columns=["sex"]),
            metrics={"sanity": ["common_rows_proportion", "data_mismatch_score"]},
        )


def test_benchmark_invalid_metric() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    score = Benchmarks.evaluate(
        [
            ("test2", "uniform_sampler", {}),
        ],
        GenericDataLoader(X, sensitive_columns=["sex"]),
        metrics={"sanity": ["invalid"]},
    )
    assert len(score["test2"]) == 0


def test_benchmark_custom_target() -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    Benchmarks.evaluate(
        [
            ("test2", "uniform_sampler", {}),
        ],
        GenericDataLoader(
            X, sensitive_columns=["sex"], target_column="sepal width (cm)"
        ),
        metrics={
            "performance": [
                "linear_model",
            ]
        },
    )


def test_benchmark_survival_analysis() -> None:
    df = load_rossi()

    with pytest.raises(ValueError):
        Benchmarks.evaluate(
            [
                ("test2", "uniform_sampler", {}),
            ],
            SurvivalAnalysisDataLoader(
                df, target_column=None, time_to_event_column="week", time_horizons=[30]
            ),
            task_type="survival_analysis",
            metrics={
                "performance": [
                    "linear_model",
                ]
            },
        )

    with pytest.raises(ValueError):
        Benchmarks.evaluate(
            [
                ("test2", "uniform_sampler", {}),
            ],
            SurvivalAnalysisDataLoader(
                df,
                target_column="arrest",
                time_to_event_column=None,
                time_horizons=[30],
            ),
            task_type="survival_analysis",
            metrics={
                "performance": [
                    "linear_model",
                ]
            },
        )

    with pytest.raises(ValueError):
        Benchmarks.evaluate(
            [
                ("test1", "uniform_sampler", {}),
            ],
            SurvivalAnalysisDataLoader(
                df,
                target_column="arrest",
                time_to_event_column="week",
                time_horizons=None,
            ),
            task_type="survival_analysis",
            metrics={
                "performance": [
                    "linear_model",
                ]
            },
        )

    score = Benchmarks.evaluate(
        [
            ("test1", "uniform_sampler", {}),
        ],
        SurvivalAnalysisDataLoader(
            df, target_column="arrest", time_to_event_column="week", time_horizons=[30]
        ),
        task_type="survival_analysis",
        metrics={
            "performance": [
                "linear_model",
            ]
        },
    )
    print(score)


def test_benchmark_workspace_cache() -> None:
    df = load_rossi()

    workspace = Path("workspace_test")
    try:
        workspace.unlink()
    except BaseException:
        pass

    Benchmarks.evaluate(
        [
            ("test1", "uniform_sampler", {}),
        ],
        SurvivalAnalysisDataLoader(
            df, target_column="arrest", time_to_event_column="week", time_horizons=[30]
        ),
        task_type="survival_analysis",
        metrics={
            "performance": [
                "linear_model",
            ]
        },
        workspace=workspace,
    )

    assert workspace.exists()
