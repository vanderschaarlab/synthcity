# third party
import pytest
from sklearn.datasets import load_diabetes, load_iris

# synthcity absolute
from synthcity.metrics.privacy import (
    evaluate_delta_presence,
    evaluate_k_anonymization,
    evaluate_kmap,
    evaluate_l_diversity,
    select_outliers,
    select_quantiles,
)
from synthcity.plugins import Plugin, Plugins


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
@pytest.mark.parametrize(
    "method", ["isolation_forests", "local_outlier_factor", "elliptic_envelope"]
)
def test_select_outliers(test_plugin: Plugin, method: str) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    outlier_index = select_outliers(X, method=method)

    assert len(outlier_index.unique()) == 2
    assert outlier_index.sum() > 0


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_select_quantiles(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)

    original_duplicates = len(X.drop_duplicates())

    quantiles = select_quantiles(X)

    assert original_duplicates > len(quantiles.drop_duplicates())


def test_evaluate_k_anonymization() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    assert evaluate_k_anonymization(X) == 18


def test_evaluate_l_diversity() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    assert evaluate_l_diversity(X, ["sex", "bmi"]) == 20


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_kmap(test_plugin: Plugin) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    test_plugin.fit(X)
    X_gen = test_plugin.generate(2 * len(X))

    assert evaluate_kmap(X, X_gen, ["sex", "bmi"]) > 18


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_delta_presence(test_plugin: Plugin) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    test_plugin.fit(X)
    X_gen = test_plugin.generate(2 * len(X))

    assert 0 < evaluate_delta_presence(X, X_gen, ["sex", "bmi"]) < 1
