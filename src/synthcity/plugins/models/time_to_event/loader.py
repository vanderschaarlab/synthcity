# stdlib
from typing import Any

# synthcity relative
from .tte_aft import WeibullAFTTimeToEvent
from .tte_coxph import CoxPHTimeToEvent
from .tte_date import DATETimeToEvent
from .tte_deephit import DeephitTimeToEvent
from .tte_rsf import RandomSurvivalForestTimeToEvent
from .tte_tenn import TENNTimeToEvent
from .tte_xgb import XGBTimeToEvent


def get_model_template(model: str) -> Any:
    defaults = {
        "tenn": TENNTimeToEvent,
        "date": DATETimeToEvent,
        "weibull_aft": WeibullAFTTimeToEvent,
        "cox_ph": CoxPHTimeToEvent,
        "random_survival_forest": RandomSurvivalForestTimeToEvent,
        "survival_xgboost": XGBTimeToEvent,
        "deephit": DeephitTimeToEvent,
    }

    if model in defaults:
        return defaults[model]

    raise RuntimeError(f"invalid model {model}")
