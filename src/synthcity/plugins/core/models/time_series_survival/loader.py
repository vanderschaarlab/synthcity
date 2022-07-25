# stdlib
from typing import Any

# synthcity relative
from .ts_surv_coxph import CoxTimeSeriesSurvival
from .ts_surv_dynamic_deephit import DynamicDeephitTimeSeriesSurvival
from .ts_surv_xgb import XGBTimeSeriesSurvival


def get_model_template(model: str) -> Any:
    defaults = {
        "cox_ph": CoxTimeSeriesSurvival,
        "dynamic_deephit": DynamicDeephitTimeSeriesSurvival,
        "ts_survival_xgboost": XGBTimeSeriesSurvival,
    }

    if model in defaults:
        return defaults[model]

    raise RuntimeError(f"invalid model {model}")
