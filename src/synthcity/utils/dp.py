# third party
import numpy as np
import pandas as pd
from diffprivlib.mechanisms import LaplaceTruncated
from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def compute_dp_marginal_distribution(
    marginal_distribution: pd.Series,
    population_size: int,
    epsilon: float = 1.0,
    delta: float = 0.0,
) -> pd.Series:
    """Convert the marginal distribution to be differential private, using the truncated Laplace mechanism, where values outside a pre-described domain are mapped to the closest point within the domain.

    Args:
        marginal_distribution: pd.Series
            The base distribution
        population_size: int
            The number of elements in the original dataset
        epsilon: float
            Privacy parameter  for the mechanism. Must be in [0, np.inf].
        delta: float
            Privacy parameter  for the mechanism. Must be in [0, 1]. Cannot be simultaneously zero with epsilon.
    """
    if epsilon <= 0:
        raise ValueError("Please provide a positive epsilon")
    # removing one record from X will decrease probability 1/n in one cell of the
    # marginal distribution and increase the probability 1/n in the remaining cells
    sensitivity = 2 / (population_size + 1e-8)
    dp_mech = LaplaceTruncated(
        epsilon=epsilon, lower=0, upper=np.iinfo(np.int32).max, sensitivity=sensitivity
    )

    dp_marginal = np.zeros_like(marginal_distribution.values)

    for i in np.arange(dp_marginal.shape[0]):
        # round counts upwards to preserve bins with noisy count between [0, 1]
        dp_marginal[i] = dp_mech.randomise(marginal_distribution.values[i])

    dp_marginal = dp_marginal / (dp_marginal.sum() + 1e-8)

    return pd.Series(dp_marginal, index=marginal_distribution.index)
