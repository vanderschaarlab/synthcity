# third party
import pandas as pd
import pytest
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics import Metrics
from synthcity.plugins import Plugins


@pytest.mark.parametrize("test_plugin", Plugins().list())
def test_basic(test_plugin: str) -> None:
    model = Plugins().get(test_plugin)

    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    model.fit(X)
    X_gen = model.generate(100)

    out = Metrics.evaluate(
        X.drop(columns=["target"]),
        X["target"],
        X_gen.drop(columns=["target"]),
        X_gen["target"],
    )

    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == set(
        [
            "mean",
            "min",
            "max",
            "median",
            "iqr",
            "stddev",
            "rounds",
            "durations",
            "errors",
            "bad_score",
            "ok_score",
        ]
    )


def test_list() -> None:
    assert set(Metrics.list().keys()) == set(
        ["privacy", "statistical", "sanity", "attacks", "detection", "performance"]
    )


@pytest.mark.parametrize(
    "metric_filter",
    [
        {"sanity": ["data_mismatch_score", "common_rows_proportion"]},
        {"sanity": ["data_mismatch_score"], "statistical": ["inverse_kl_divergence"]},
    ],
)
def test_metric_filter(metric_filter: dict) -> None:
    model = Plugins().get("dummy_sampler")

    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    model.fit(X)
    X_gen = model.generate(100)

    out = Metrics.evaluate(
        X.drop(columns=["target"]),
        X["target"],
        X_gen.drop(columns=["target"]),
        X_gen["target"],
        metrics=metric_filter,
    )

    expected_index = [
        f"{category}.{metric}"
        for category in metric_filter
        for metric in metric_filter[category]
    ]
    assert set(list(out.index)) == set(expected_index)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == set(
        [
            "mean",
            "min",
            "max",
            "median",
            "iqr",
            "stddev",
            "rounds",
            "durations",
            "errors",
            "bad_score",
            "ok_score",
        ]
    )
