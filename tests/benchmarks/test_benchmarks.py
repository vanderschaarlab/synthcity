# stdlib
import hashlib
import json
import platform
from copy import copy
from pathlib import Path
from typing import Any, List

# third party
import pytest
from lifelines.datasets import load_rossi
from sklearn.datasets import load_diabetes, load_iris

# synthcity absolute
from synthcity.benchmark import Benchmarks
from synthcity.benchmark.utils import get_json_serializable_kwargs
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import (
    DataLoader,
    GenericDataLoader,
    SurvivalAnalysisDataLoader,
)
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


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


def test_benchmark_augmentation() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    scores = Benchmarks.evaluate(
        [
            ("test1", "marginal_distributions", {}),
            ("test2", "dummy_sampler", {}),
        ],
        GenericDataLoader(X, sensitive_columns=["sex"]),
        metrics={
            "performance": [
                "linear_model_augmentation",
                "mlp_augmentation",
                "xgb_augmentation",
            ]
        },
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
    Benchmarks.print(score)

    score = Benchmarks.evaluate(
        [
            ("test1", "marginal_distributions", {}),
            ("test2", "dummy_sampler", {}),
        ],
        SurvivalAnalysisDataLoader(
            df,
            target_column="arrest",
            fairness_column="age",
            time_to_event_column="week",
            time_horizons=[30],
        ),
        task_type="survival_analysis",
        metrics={
            "performance": [
                "linear_model",
                "linear_model_augmentation",
            ]
        },
    )
    Benchmarks.print(score)


def test_benchmark_workspace_cache() -> None:
    df = load_rossi()

    workspace = Path("workspace_test")
    try:
        workspace.unlink()
    except BaseException:
        pass

    X = SurvivalAnalysisDataLoader(
        df,
        target_column="arrest",
        fairness_column="age",
        time_to_event_column="week",
        time_horizons=[30],
    )

    testcase = "test1"
    plugin = "uniform_sampler"
    kwargs = {"workspace": Path("workspace_test")}

    kwargs_hash = ""
    if len(kwargs) > 0:
        serializable_kwargs = get_json_serializable_kwargs(kwargs)
        kwargs_hash_raw = json.dumps(serializable_kwargs, sort_keys=True).encode()
        hash_object = hashlib.sha256(kwargs_hash_raw)
        kwargs_hash = hash_object.hexdigest()

    augmentation_arguments = {
        "augmentation_rule": "equal",
        "strict_augmentation": False,
        "ad_hoc_augment_vals": None,
    }
    augmentation_arguments_hash_raw = json.dumps(
        copy(augmentation_arguments), sort_keys=True
    ).encode()
    augmentation_hash_object = hashlib.sha256(augmentation_arguments_hash_raw)
    augmentation_hash = augmentation_hash_object.hexdigest()

    experiment_name = X.hash()
    repeats = 3

    Benchmarks.evaluate(
        [
            (testcase, plugin, kwargs),
        ],
        X,
        task_type="survival_analysis",
        metrics={
            "performance": [
                "linear_model_augmentation",
            ]
        },
        repeats=repeats,
        workspace=workspace,
        augmented_reuse_if_exists=False,
        synthetic_reuse_if_exists=False,
    )

    assert workspace.exists()

    for repeat in range(repeats):
        X_syn_cache_file = (
            workspace
            / f"{experiment_name}_{testcase}_{plugin}_{kwargs_hash}_{platform.python_version()}_{repeat}.bkp"
        )
        generator_file = (
            workspace
            / f"{experiment_name}_{testcase}_{plugin}_{kwargs_hash}_{platform.python_version()}_generator_{repeat}.bkp"
        )
        X_augment_cache_file = (
            workspace
            / f"{experiment_name}_{testcase}_{plugin}_augmentation_{augmentation_hash}_{kwargs_hash}_{platform.python_version()}_{repeat}.bkp"
        )

        augment_generator_file = (
            workspace
            / f"{experiment_name}_{testcase}_{plugin}_augmentation_{augmentation_hash}_{kwargs_hash}_{platform.python_version()}_generator_{repeat}.bkp"
        )

        assert X_syn_cache_file.exists()
        assert generator_file.exists()

        assert X_augment_cache_file.exists()
        assert augment_generator_file.exists()


def test_benchmark_added_plugin() -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    class DummyCopyDataPlugin(Plugin):
        """Dummy plugin for debugging."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)

        @staticmethod
        def name() -> str:
            return "copy_data"

        @staticmethod
        def type() -> str:
            return "debug"

        @staticmethod
        def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
            return []

        def _fit(
            self, X: DataLoader, *args: Any, **kwargs: Any
        ) -> "DummyCopyDataPlugin":
            self.features_count = X.shape[1]
            self.X = X
            return self

        def _generate(
            self, count: int, syn_schema: Schema, **kwargs: Any
        ) -> DataLoader:
            return self.X.sample(count)

    generators = Plugins()
    # Add the new plugin to the collection
    generators.add("copy_data", DummyCopyDataPlugin)

    score = Benchmarks.evaluate(
        [
            ("copy_data", "copy_data", {}),
        ],
        GenericDataLoader(X, target_column="target"),
        metrics={
            "performance": [
                "linear_model",
            ]
        },
    )
    assert "copy_data" in score
