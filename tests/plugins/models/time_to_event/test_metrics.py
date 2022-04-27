# third party
import pandas as pd

# synthcity absolute
from synthcity.plugins.models.time_to_event.metrics import (
    expected_time_error,
    ranking_error,
    rush_error,
)


def test_expected_time_error() -> None:
    T = pd.Series([10])
    E = pd.Series([1], index=T.index)
    pred_T = pd.Series([11], index=T.index)
    assert expected_time_error(T, E, pred_T) == 1

    T = pd.Series([10])
    E = pd.Series([0], index=T.index)
    pred_T = pd.Series([9], index=T.index)
    assert expected_time_error(T, E, pred_T) == 1

    T = pd.Series([10, 10])
    E = pd.Series([0, 1], index=T.index)
    pred_T = pd.Series([9, 9], index=T.index)
    assert expected_time_error(T, E, pred_T) == 2


def test_ranking_error() -> None:
    T = pd.Series([10, 12])
    E = pd.Series([0, 0], index=T.index)
    pred_T = pd.Series([11, 9], index=T.index)
    assert ranking_error(T, E, pred_T) == 0

    T = pd.Series([10, 12])
    E = pd.Series([1, 0], index=T.index)
    pred_T = pd.Series([11, 9], index=T.index)
    assert ranking_error(T, E, pred_T) == 1


def test_rush_error() -> None:
    T = pd.Series([10, 12])
    pred_T = pd.Series([11, 9], index=T.index)
    assert rush_error(T, pred_T) == 0.5

    T = pd.Series([10, 12])
    pred_T = pd.Series([11, 13], index=T.index)
    assert rush_error(T, pred_T) == 0
