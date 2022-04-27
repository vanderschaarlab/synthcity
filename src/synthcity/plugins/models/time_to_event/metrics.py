# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.metrics import mean_squared_error


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def expected_time_error(T: pd.Series, E: pd.Series, pred_T: pd.Series) -> float:
    """
    Returns an evaluation error for both observed and censored measurements.

    Errors:
        * For lab measurements(E == 1): RMSE(predicted, T)
        * For censored measurements(E == 0): RMSE(predicted, T), if predicted < T.
    """
    err = 0
    if (E == 0).sum() > 0:
        censored_err = T[E == 0] - pred_T[E == 0]
        censored_err[censored_err < 0] = 0
        censored_err = np.sqrt(np.mean(censored_err**2))

        err += censored_err

    if (E == 1).sum() > 0:
        obs_error = np.sqrt(mean_squared_error(T[E == 1], pred_T[E == 1]))
        err += obs_error

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
def rush_error(T: pd.Series, pred_T: pd.Series) -> float:
    """
    Returns the proportions of time-to-event predictions before the actual observed/censoring time.
    """

    return (T > pred_T).sum() / len(pred_T)
