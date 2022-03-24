# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd

# synthcity absolute
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class UniformSamplerPlugin(Plugin):
    """Dummy plugin for debugging.

    Example:
        >>> from synthcity.plugins import Plugins
        >>> plugin = Plugins().get("uniform_sampler")
        >>> from sklearn.datasets import load_iris
        >>> X = load_iris()
        >>> plugin.fit(X)
        >>> plugin.generate()
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "uniform_sampler"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        return []

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "UniformSamplerPlugin":
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        X_rnd = pd.DataFrame(
            np.zeros((count, len(self.schema().features()))),
            columns=self.schema().features(),
        )

        for feature in syn_schema:
            sample = syn_schema[feature].sample(count=count)
            X_rnd[feature] = sample
        return X_rnd


plugin = UniformSamplerPlugin
