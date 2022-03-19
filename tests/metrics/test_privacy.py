# third party
import pytest
from sklearn.datasets import load_breast_cancer, load_iris

# synthcity absolute
from synthcity.metrics.privacy import (
    evaluate_k_anonimity,
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
    outlier_index = select_outliers(X, y, method=method)

    assert len(outlier_index.unique()) == 2
    assert outlier_index.sum() > 0


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_select_quantiles(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)

    original_duplicates = len(X.drop_duplicates())

    quantiles = select_quantiles(X)

    assert original_duplicates > len(quantiles.drop_duplicates())


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_k_anonimity(test_plugin: Plugin) -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    assert evaluate_k_anonimity(X, k=1) is True
    assert evaluate_k_anonimity(X, k=2) is False
