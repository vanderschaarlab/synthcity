# third party
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from pydantic import validate_arguments
from sklearn.metrics import mean_absolute_error, mean_squared_error


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def expected_time_error(
    T: pd.Series,
    E: pd.Series,
    pred_T: pd.Series,
    nc_weight: int = 10,
    c_weight: int = 1,
    metric: str = "l1",
) -> float:
    """
    Returns an evaluation error for both observed and censored measurements.

    Errors:
        * For lab measurements(E == 1): distance(predicted, T)
        * For censored measurements(E == 0): distance(predicted, T), if predicted < T.
    """

    def _distance(lhs: pd.Series, rhs: pd.Series) -> float:
        if len(lhs) == 0:
            return 0

        if len(lhs) != len(rhs):
            return 0

        if metric == "l1":
            return mean_absolute_error(lhs, rhs)
        elif metric == "l2":
            return np.sqrt(mean_squared_error(lhs, rhs))
        else:
            raise ValueError(f"unknown evaluation metric {metric}")

    err = 0
    if (E == 0).sum() > 0:
        lhs = T[E == 0]
        rhs = pred_T[E == 0]

        lhs = lhs[lhs > rhs]
        rhs = rhs[lhs.index]

        censored_err = _distance(lhs, rhs)
        err += c_weight * censored_err / (T.max() + 1e-8)

    if (E == 1).sum() > 0:
        obs_error = _distance(T[E == 1], pred_T[E == 1])
        err += nc_weight * obs_error / (T.max() + 1e-8)

    return err


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def expected_time_error_l1(
    T: pd.Series,
    E: pd.Series,
    pred_T: pd.Series,
    nc_weight: int = 10,
    c_weight: int = 1,
) -> float:
    """
    Returns an evaluation error for both observed and censored measurements.

    Errors:
        * For lab measurements(E == 1): L1(predicted, T)
        * For censored measurements(E == 0): L1(predicted, T), if predicted < T.
    """
    err = 0
    if (E == 0).sum() > 0:
        censored_err = T[E == 0] - pred_T[E == 0]
        censored_err[censored_err < 0] = 0
        censored_err = np.mean(censored_err)

        err += c_weight * censored_err / (T.max() + 1e-8)

    if (E == 1).sum() > 0:
        obs_error = np.sqrt(mean_squared_error(T[E == 1], pred_T[E == 1]))
        err += nc_weight * obs_error / (T.max() + 1e-8)

    return err


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ranking_error(T: pd.Series, E: pd.Series, pred_T: pd.Series) -> float:
    """
    Returns an error for the out-of-order predictions.

    Errors:
        - For every value t in T, we check if T[E == 1] < t and pred_T[E == 1] < t have the same indexes.
    """
    if (E == 1).sum() == 0:
        return 0

    rank_errs = 0
    for idx in range(len(T)):
        actual_order = (T[E == 1] < T.iloc[idx]).astype(int)
        pred_order = (pred_T[E == 1] < pred_T.iloc[idx]).astype(int)
        rank_errs += np.sqrt(mean_squared_error(pred_order, actual_order))

    return rank_errs


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def c_index(T: pd.Series, E: pd.Series, pred_T: pd.Series) -> float:
    """
    Returns the cindex.
    """
    try:
        return concordance_index(
            event_times=T, predicted_scores=pred_T, event_observed=E
        )
    except BaseException:
        return 0


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def rush_error(T: pd.Series, pred_T: pd.Series) -> float:
    """
    Returns the proportions of time-to-event predictions before the actual observed/censoring time.
    """

    return (T > pred_T).sum() / len(pred_T)
