# third party
import pytest
from sklearn.datasets import load_diabetes

# synthcity absolute
from synthcity.benchmark import Benchmarks


def test_benchmark_sanity() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    scores = Benchmarks.evaluate(
        [
            "dummy_sampler",
            "random_noise",
        ],
        X,
        y,
        sensitive_columns=["sex"],
        metrics={"sanity": ["common_rows_proportion", "data_mismatch_score"]},
    )

    Benchmarks.print(scores)


def test_benchmark_invalid_plugin() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    with pytest.raises(ValueError):
        Benchmarks.evaluate(
            [
                "invalid",
                "random_noise",
            ],
            X,
            y,
            sensitive_columns=["sex"],
            metrics={"sanity": ["common_rows_proportion", "data_mismatch_score"]},
        )


def test_benchmark_invalid_metric() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    score = Benchmarks.evaluate(
        [
            "random_noise",
        ],
        X,
        y,
        sensitive_columns=["sex"],
        metrics={"sanity": ["invalid"]},
    )
    assert len(score["random_noise"]) == 0
