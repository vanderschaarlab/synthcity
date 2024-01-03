# stdlib
from typing import Any, Tuple

# third party
import pandas as pd

# synthcity relative
from .surv_aft import WeibullAFTSurvivalAnalysis
from .surv_coxph import CoxPHSurvivalAnalysis
from .surv_deephit import DeephitSurvivalAnalysis
from .surv_xgb import XGBSurvivalAnalysis


def generate_dataset_for_horizon(
    X: pd.DataFrame, T: pd.DataFrame, Y: pd.DataFrame, horizon_days: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate the dataset at a certain time horizon. Useful for classifiers.

    Args:
        X: pd.DataFrame, the feature set
        T: pd.DataFrame, days to event or censoring
        Y: pd.DataFrame, outcome or censoring
        horizon_days: int, days to the expected horizon

    Returns:
        X: the feature set for that horizon
        T: days to event or censoring
        Y: Outcome or censoring

    """

    event_horizon = ((Y == 1) & (T <= horizon_days)) | ((Y == 0) & (T > horizon_days))
    censored_event_horizon = (Y == 1) & (T > horizon_days)

    X_horizon = X[event_horizon].reset_index(drop=True)
    X_horizon_cens = X[censored_event_horizon].reset_index(drop=True)

    Y_horizon = Y[event_horizon].reset_index(drop=True)
    Y_horizon_cens = 1 - Y[censored_event_horizon].reset_index(drop=True)

    T_horizon = T[event_horizon].reset_index(drop=True)
    T_horizon_cens = T[censored_event_horizon].reset_index(drop=True)

    return (
        pd.concat([X_horizon, X_horizon_cens], ignore_index=True),
        pd.concat([T_horizon, T_horizon_cens], ignore_index=True),
        pd.concat([Y_horizon, Y_horizon_cens], ignore_index=True),
    )


def get_model_template(model: str) -> Any:
    defaults = {
        "weibull_aft": WeibullAFTSurvivalAnalysis,
        "cox_ph": CoxPHSurvivalAnalysis,
        "survival_xgboost": XGBSurvivalAnalysis,
        "deephit": DeephitSurvivalAnalysis,
    }

    if model in defaults:
        return defaults[model]

    raise RuntimeError(f"invalid model {model}")
