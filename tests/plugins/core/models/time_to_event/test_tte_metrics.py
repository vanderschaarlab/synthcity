# third party
import numpy as np
import pandas as pd
import pytest

# synthcity absolute
from synthcity.plugins.core.models.time_to_event.metrics import (
    c_index,
    expected_time_error,
    ranking_error,
    rush_error,
)


@pytest.mark.parametrize(
    "metric, results",
    [
        ("l1", (1, 0.099, 1)),
        ("l2", (1, 0.099, 1)),
    ],
)
def test_expected_time_error(metric: str, results: tuple) -> None:
    T = pd.Series([10])
    E = pd.Series([1], index=T.index)
    pred_T = pd.Series([11], index=T.index)
    assert np.abs(expected_time_error(T, E, pred_T, metric=metric) - results[0]) < 0.1

    T = pd.Series([10])
    E = pd.Series([0], index=T.index)
    pred_T = pd.Series([9], index=T.index)
    assert np.abs(expected_time_error(T, E, pred_T, metric=metric) - results[1]) < 0.1

    T = pd.Series([10, 10])
    E = pd.Series([0, 1], index=T.index)
    pred_T = pd.Series([9, 9], index=T.index)
    assert np.abs(expected_time_error(T, E, pred_T, metric=metric) - results[2]) < 0.1


def test_ranking_error() -> None:
    T = pd.Series([10, 12])
    E = pd.Series([0, 0], index=T.index)
    pred_T = pd.Series([11, 9], index=T.index)
    assert ranking_error(T, E, pred_T) == 0

    T = pd.Series([10, 12])
    E = pd.Series([1, 0], index=T.index)
    pred_T = pd.Series([11, 9], index=T.index)
    assert ranking_error(T, E, pred_T) == 1


def test_c_index() -> None:
    T = pd.Series([10, 12])
    E = pd.Series([0, 0], index=T.index)
    pred_T = pd.Series([11, 9], index=T.index)
    assert c_index(T, E, pred_T) == 0

    T = pd.Series([10, 12])
    E = pd.Series([1, 0], index=T.index)
    pred_T = pd.Series([11, 19], index=T.index)
    assert c_index(T, E, pred_T) == 1


def test_rush_error() -> None:
    T = pd.Series([10, 12])
    pred_T = pd.Series([11, 9], index=T.index)
    assert rush_error(T, pred_T) == 0.5

    T = pd.Series([10, 12])
    pred_T = pd.Series([11, 13], index=T.index)
    assert rush_error(T, pred_T) == 0
