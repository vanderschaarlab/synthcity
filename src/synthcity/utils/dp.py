# stdlib
from sys import maxsize
from typing import Any

# third party
import numpy as np
import pandas as pd
from diffprivlib.mechanisms import LaplaceTruncated
from pydantic import validate_arguments
from thomas.core import CPT, JPT, Factor

# synthcity relative
from .statistics import (
    _normalize_cpt,
    _normalize_distribution,
    contingency_table,
    joint_distribution,
    marginal_distribution,
)


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


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def dp_contingency_table(data: pd.DataFrame, epsilon: float) -> Factor:
    """Compute differentially private contingency table of input data"""
    contingency_table_ = contingency_table(data)

    # if we remove one record from X the count in one cell decreases by 1 while the rest stays the same.
    sensitivity = 1
    dp_mech = LaplaceTruncated(
        epsilon=epsilon, lower=0, upper=maxsize, sensitivity=sensitivity
    )

    contingency_table_values = contingency_table_.values.flatten()
    dp_contingency_table = np.zeros_like(contingency_table_values)
    for i in np.arange(dp_contingency_table.shape[0]):
        # round counts upwards to preserve bins with noisy count between [0, 1]
        dp_contingency_table[i] = np.ceil(
            dp_mech.randomise(contingency_table_values[i])
        )

    return Factor(dp_contingency_table, states=contingency_table_.states)


def dp_marginal_distribution(data: pd.DataFrame, epsilon: float) -> Factor:
    """Compute differentially private marginal distribution of input data"""
    marginal_ = marginal_distribution(data)

    # removing one record from X will decrease probability 1/n in one cell of the
    # marginal distribution and increase the probability 1/n in the remaining cells
    sensitivity = 2 / data.shape[0]
    dp_mech = LaplaceTruncated(
        epsilon=epsilon, lower=0, upper=maxsize, sensitivity=sensitivity
    )

    dp_marginal = np.zeros_like(marginal_.values)

    for i in np.arange(dp_marginal.shape[0]):
        # round counts upwards to preserve bins with noisy count between [0, 1]
        dp_marginal[i] = dp_mech.randomise(marginal_.values[i])

    dp_marginal = _normalize_distribution(dp_marginal)
    return Factor(dp_marginal, states=marginal_.states)


def dp_joint_distribution(data: pd.DataFrame, epsilon: float) -> Factor:
    """Compute differentially private joint distribution of input data"""
    joint_distribution_ = joint_distribution(data)

    # removing one record from X will decrease probability 1/n in one cell of the
    # joint distribution and increase the probability 1/n in the remaining cells
    sensitivity = 2 / data.shape[0]
    dp_mech = LaplaceTruncated(
        epsilon=epsilon, lower=0, upper=maxsize, sensitivity=sensitivity
    )

    joint_distribution_values = joint_distribution_.values.flatten()
    dp_joint_distribution_ = np.zeros_like(joint_distribution_values)

    for i in np.arange(dp_joint_distribution_.shape[0]):
        dp_joint_distribution_[i] = dp_mech.randomise(joint_distribution_values[i])

    dp_joint_distribution_ = _normalize_distribution(dp_joint_distribution_)
    return JPT(dp_joint_distribution_, states=joint_distribution_.states)


def dp_conditional_distribution(
    data: pd.DataFrame, epsilon: float, conditioned: Any = None
) -> CPT:
    """Compute differentially private conditional distribution of input data
    Inferred from marginal or joint distribution"""
    # if only one columns (series or dataframe), i.e. no conditioning columns
    if len(data.squeeze().shape) == 1:
        dp_distribution = dp_marginal_distribution(data, epsilon=epsilon)
    else:
        dp_distribution = dp_joint_distribution(data, epsilon=epsilon)
    cpt = CPT(dp_distribution, conditioned=conditioned)

    # normalize if cpt has conditioning columns
    if cpt.conditioning:
        cpt = _normalize_cpt(cpt)
    return cpt
