# stdlib
from typing import Any, Dict, List

# third party
import optuna

# synthcity absolute
import synthcity.plugins.core.distribution as D
from synthcity.plugins.core.distribution import Distribution


def _dispatch(distribution: Distribution) -> optuna.distributions.BaseDistribution:
    if isinstance(distribution, D.FloatDistribution):
        return optuna.distributions.UniformDistribution(
            distribution.low, distribution.high
        )
    elif isinstance(distribution, D.LogDistribution):
        return optuna.distributions.LogUniformDistribution(
            distribution.low, distribution.high
        )
    elif isinstance(distribution, D.IntegerDistribution):
        return optuna.distributions.IntUniformDistribution(
            distribution.low, distribution.high, distribution.step
        )
    elif isinstance(distribution, D.LogIntDistribution):
        return optuna.distributions.IntLogUniformDistribution(
            distribution.low, distribution.high
        )
    elif isinstance(distribution, D.CategoricalDistribution):
        return optuna.distributions.CategoricalDistribution(distribution.choices)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def suggest(trial: optuna.Trial, distribution: Distribution) -> Any:
    return trial._suggest(name=distribution.name, distribution=_dispatch(distribution))


def suggest_all(trial: optuna.Trial, distributions: List[Distribution]) -> Dict:
    return {
        distribution.name: suggest(trial, distribution)
        for distribution in distributions
    }
