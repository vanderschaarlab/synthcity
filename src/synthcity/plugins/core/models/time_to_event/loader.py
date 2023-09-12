# stdlib
from typing import Any

# synthcity relative
from .tte_aft import WeibullAFTTimeToEvent
from .tte_coxph import CoxPHTimeToEvent
from .tte_date import DATETimeToEvent
from .tte_deephit import DeephitTimeToEvent
from .tte_survival_function_regression import SurvivalFunctionTimeToEvent
from .tte_tenn import TENNTimeToEvent
from .tte_xgb import XGBTimeToEvent


def get_model_template(model: str) -> Any:
    defaults = {
        "tenn": TENNTimeToEvent,
        "date": DATETimeToEvent,
        "weibull_aft": WeibullAFTTimeToEvent,
        "cox_ph": CoxPHTimeToEvent,
        "survival_xgboost": XGBTimeToEvent,
        "deephit": DeephitTimeToEvent,
        "survival_function_regression": SurvivalFunctionTimeToEvent,
    }

    if model in defaults:
        return defaults[model]

    raise RuntimeError(f"invalid model {model}")
