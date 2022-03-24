# third party
import pandas as pd
import pytest

# synthcity absolute
from synthcity.utils.dp import compute_dp_marginal_distribution


@pytest.mark.parametrize("epsilon", [1, 10, 100])
@pytest.mark.parametrize("delta", [0, 0.5, 1])
@pytest.mark.parametrize("population_size", [1, 10, 1000])
def test_compute_dp_marginal_distribution(
    epsilon: float, delta: float, population_size: int
) -> None:
    data = pd.Series([1, 2, 3], index=[1, 2, 3])
    data /= data.sum()

    dist = compute_dp_marginal_distribution(data, population_size, epsilon=epsilon)

    assert (data.values != dist.values).any()
