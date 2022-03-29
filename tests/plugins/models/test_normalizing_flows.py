# third party
import pytest
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.plugins.models import NormalizingFlows


def test_nf_sanity() -> None:
    model = NormalizingFlows(
        n_iter=1001,
        n_iter_print=101,
        n_hidden_layers=6,
        n_hidden_units=11,
        batch_size=101,
        base_distribution="diagonal_normal",
        linear_transform_type="permutation",
        base_transform_type="affine-coupling",
    )

    assert model.n_iter == 1001
    assert model.n_iter_print == 101
    assert model.n_hidden_layers == 6
    assert model.n_hidden_units == 11
    assert model.batch_size == 101
    assert model.base_distribution == "diagonal_normal"
    assert model.linear_transform_type == "permutation"
    assert model.base_transform_type == "affine-coupling"


@pytest.mark.parametrize("base_distribution", ["standard_normal", "diagonal_normal"])
@pytest.mark.parametrize("linear_transform_type", ["lu", "permutation", "svd"])
@pytest.mark.parametrize(
    "base_transform_type",
    [
        "affine-coupling",
        "quadratic-coupling",
        "rq-coupling",
        "affine-autoregressive",
        "quadratic-autoregressive",
        "rq-autoregressive",
    ],
)
def test_nf_fit(
    base_distribution: str, linear_transform_type: str, base_transform_type: str
) -> None:
    X, _ = load_iris(return_X_y=True)

    NormalizingFlows(
        n_iter=10,
        base_distribution=base_distribution,
        linear_transform_type=linear_transform_type,
        base_transform_type=base_transform_type,
    ).fit(X)
