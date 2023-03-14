# third party
import pandas as pd
import pytest
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.benchmark.utils import augment_data
from synthcity.metrics import Metrics, WeightedMetrics
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader


@pytest.mark.parametrize("test_plugin", ["dummy_sampler", "marginal_distributions"])
def test_basic(test_plugin: str) -> None:
    model = Plugins().get(test_plugin)

    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    Xraw = GenericDataLoader(X, target_column="target")

    model.fit(Xraw)
    X_gen = model.generate(100)

    out = Metrics.evaluate(
        Xraw,
        X_gen,
        metrics={"sanity": ["common_rows_proportion"]},
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
    X["target"] = X["target"].astype(str)

    Xraw = GenericDataLoader(X, target_column="target")

    model.fit(Xraw)
    X_gen = model.generate(100)

    out = Metrics.evaluate(
        Xraw,
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
    X["target"] = X["target"].astype(str)

    Xraw = GenericDataLoader(X, target_column="target")

    model.fit(Xraw)
    X_gen = model.generate(100)

    out = Metrics.evaluate(Xraw, X_gen, metrics={"performance": "linear_model"})

    assert "performance.linear_model.syn_id" in out.index
    assert "performance.linear_model.syn_ood" in out.index


@pytest.mark.parametrize("test_plugin", ["dummy_sampler"])
def test_weighted_metric(test_plugin: str) -> None:
    model = Plugins().get(test_plugin)

    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    Xraw = GenericDataLoader(X, target_column="target")

    model.fit(Xraw)
    X_gen = model.generate(100)

    score = WeightedMetrics(
        [
            ("sanity", "common_rows_proportion"),
            ("sanity", "data_mismatch"),
        ],
        weights=[0.5, 0.5],
    ).evaluate(
        Xraw,
        X_gen,
    )

    assert isinstance(score, (float, int))

    # invalid metric
    with pytest.raises(ValueError):
        WeightedMetrics([("sanity", "fake")], [1])

    # invalid weights
    with pytest.raises(ValueError):
        WeightedMetrics([("sanity", "common_rows_proportion")], [0.5, 0.5])

    # different direction
    with pytest.raises(ValueError):
        WeightedMetrics(
            [("sanity", "common_rows_proportion"), ("performance", "xgb")], [0.5, 0.5]
        )


@pytest.mark.parametrize(
    "fairness_column, rule, strict, ad_hoc_vals",
    [
        ("sepal length (cm)", "equal", True, {}),
        ("sepal length (cm)", "equal", False, {}),
        ("sepal length (cm)", "log", True, {}),
        ("sepal length (cm)", "log", False, {}),
        (
            "sepal length (cm)",
            "ad-hoc",
            True,
            {k: 10 for k in [4.6, 5.0, 5.4, 4.4, 4.8, 7.4, 7.9]},
        ),
        (
            "sepal length (cm)",
            "ad-hoc",
            False,
            {k: 10 for k in [4.6, 5.0, 5.4, 4.4, 4.8, 7.4, 7.9]},
        ),
        pytest.param(
            "sepal length (cm)",
            "ad-hoc",
            True,
            {k: 10 for k in [-1, 10000]},
            marks=pytest.mark.xfail,
        ),
    ],
)
def test_augmentation(
    fairness_column: str, rule: str, strict: bool, ad_hoc_vals: dict
) -> None:
    augment_generator = Plugins().get("marginal_distributions")

    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    X["target"] = X["target"].astype(str)

    Xraw = GenericDataLoader(X, target_column="target", fairness_column=fairness_column)

    augment_generator.fit(Xraw, cond=Xraw[fairness_column])

    X_augmented = augment_data(
        Xraw,
        augment_generator,
        rule=rule,
        strict=strict,
        ad_hoc_augment_vals=ad_hoc_vals,
    )
    assert len(Xraw) < len(X_augmented)

    X_gen = augment_generator.generate(100)

    out = Metrics.evaluate(
        Xraw, X_gen, X_augmented, metrics={"performance": "linear_model_augmentation"}
    )
    assert "performance.linear_model_augmentation.aug_ood" in out.index
