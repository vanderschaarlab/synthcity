# third party
import pandas as pd
import pytest
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics import Metrics
from synthcity.plugins import Plugins


@pytest.mark.parametrize("test_plugin", ["dummy_sampler", "marginal_distributions"])
def test_basic(test_plugin: str) -> None:
    model = Plugins().get(test_plugin)

    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    model.fit(X)
    X_gen = model.generate(100)

    out = Metrics.evaluate(
        X,
        X_gen,
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
            "direction",
        ]
    )


def test_list() -> None:
    assert set(Metrics.list().keys()) == set(
        [
            "privacy",
            "stats",
            "sanity",
            "detection",
            "performance",
        ]
    )


@pytest.mark.parametrize(
    "metric_filter",
    [
        {"sanity": ["data_mismatch", "common_rows_proportion"]},
    ],
)
def test_metric_filter(metric_filter: dict) -> None:
    model = Plugins().get("marginal_distributions")

    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    model.fit(X)
    X_gen = model.generate(100)

    out = Metrics.evaluate(
        X,
        X_gen,
        metrics=metric_filter,
    )

    expected_index = [
        f"{category}.{metric}.score"
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
            "direction",
        ]
    )


@pytest.mark.parametrize(
    "target",
    [
        None,
        "target",
        "sepal width (cm)",
    ],
)
def test_custom_label(target: str) -> None:
    model = Plugins().get("marginal_distributions")

    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    model.fit(X)
    X_gen = model.generate(100)

    out = Metrics.evaluate(
        X, X_gen, target_column=target, metrics={"performance": "linear_model"}
    )

    assert "performance.linear_model.syn" in out.index
