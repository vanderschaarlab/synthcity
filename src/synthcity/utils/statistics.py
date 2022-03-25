# stdlib
from typing import Union

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from thomas.core import CPT, Factor


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def contingency_table(data: pd.DataFrame) -> Factor:
    return Factor.from_data(data)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def joint_distribution(data: pd.DataFrame) -> Factor:
    """Get joint distribution by normalizing contingency table"""
    return contingency_table(data).normalize()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def marginal_distribution(data: Union[pd.DataFrame, pd.Series]) -> Factor:
    assert len(data.squeeze().shape) == 1, "data can only consist of a single column"
    # converts single column dataframe to series
    data = data.squeeze()

    marginal = data.value_counts(normalize=True, dropna=False)
    states = {data.name: marginal.index.tolist()}
    return Factor(marginal, states=states)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def uniform_distribution(data: Union[pd.DataFrame, pd.Series]) -> Factor:
    assert len(data.squeeze().shape) == 1, "data can only consist of a single column"
    # converts single column dataframe to series
    data = data.squeeze()
    n_unique = data.nunique(dropna=False)
    uniform = np.full(n_unique, 1 / n_unique)
    states = {data.name: data.unique().tolist()}
    return Factor(uniform, states=states)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def compute_distribution(data: Union[pd.DataFrame, pd.Series]) -> Factor:
    """ "Draws a marginal or joint distribution depending on the number of input dimensions"""
    if len(data.squeeze().shape) == 1:
        return marginal_distribution(data)
    else:
        return joint_distribution(data)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def _normalize_distribution(distribution: np.ndarray) -> np.ndarray:
    """Check whether probability distribution sums to 1"""
    distribution = _check_all_zero(distribution)
    distribution = distribution / distribution.sum()
    return distribution


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def _check_all_zero(distribution: np.ndarray) -> np.ndarray:
    """In case distribution contains only zero values due to DP noise, convert to uniform"""
    if not np.any(distribution):
        distribution = np.repeat(1 / len(distribution), repeats=len(distribution))
    return distribution


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def _normalize_cpt(cpt: pd.DataFrame) -> CPT:
    """normalization of cpt with option to fill missing values with uniform distribution"""
    # convert to series as normalize does not work with thomas cpts
    series = cpt.as_series()
    series_norm_full = series / series.unstack().sum(axis=1)
    # fill missing combinations with uniform distribution
    uniform_prob = 1 / len(cpt.states[cpt.conditioned[-1]])
    series_norm_full = series_norm_full.fillna(uniform_prob)
    return CPT(series_norm_full, cpt.states)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def cardinality(X: pd.Series) -> int:
    """Compute cardinality of input data"""
    return np.prod(X.nunique(dropna=False))
