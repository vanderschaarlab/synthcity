# stdlib
from typing import Any

# synthcity relative
from .surv_aft import WeibullAFTSurvivalAnalysis
from .surv_coxph import CoxPHSurvivalAnalysis
from .tte_deephit import DeephitSurvivalAnalysis
from .tte_xgb import XGBSurvivalAnalysis


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
