# third party
import pytest
from sklearn.datasets import load_diabetes

# synthcity absolute
from synthcity.benchmark import Benchmarks


def test_benchmark_sanity() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    scores = Benchmarks.evaluate(
        [
            "marginal_distributions",
            "uniform_sampler",
        ],
        X,
        sensitive_columns=["sex"],
        metrics={"sanity": ["common_rows_proportion", "data_mismatch_score"]},
    )

    Benchmarks.print(scores)


def test_benchmark_invalid_plugin() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    with pytest.raises(ValueError):
        Benchmarks.evaluate(
            [
                "invalid",
                "uniform_sampler",
            ],
            X,
            sensitive_columns=["sex"],
            metrics={"sanity": ["common_rows_proportion", "data_mismatch_score"]},
        )


def test_benchmark_invalid_metric() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    score = Benchmarks.evaluate(
        [
            "uniform_sampler",
        ],
        X,
        sensitive_columns=["sex"],
        metrics={"sanity": ["invalid"]},
    )
    assert len(score["uniform_sampler"]) == 0
