# stdlib
from typing import Any, Dict, List

# third party
import optuna

# synthcity absolute
import synthcity.plugins.core.distribution as D


def suggest(trial: optuna.Trial, dist: D.Distribution) -> Any:
    if isinstance(dist, D.FloatDistribution):
        return trial.suggest_float(dist.name, dist.low, dist.high)
    elif isinstance(dist, D.LogDistribution):
        return trial.suggest_float(dist.name, dist.low, dist.high, log=True)
    elif isinstance(dist, D.IntegerDistribution):
        return trial.suggest_int(dist.name, dist.low, dist.high, dist.step)
    elif isinstance(dist, D.IntLogDistribution):
        # ! does not handle step yet
        return trial.suggest_int(dist.name, dist.low, dist.high, log=True)
    elif isinstance(dist, D.CategoricalDistribution):
        return trial.suggest_categorical(dist.name, dist.choices)
    # ! the modification cannot be reflected in study.best_params
    # elif isinstance(dist, D.DatetimeDistribution):
    #     high = (dist.high - dist.low) / dist.step
    #     s = trial.suggest_float(dist.name, 0, high)
    #     return dist.low + dist.step * s
    else:
        raise ValueError(f"Unknown dist: {dist}")


def suggest_all(trial: optuna.Trial, distributions: List[D.Distribution]) -> Dict:
    return {dist.name: suggest(trial, dist) for dist in distributions}
