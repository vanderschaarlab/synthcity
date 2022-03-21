# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd

# synthcity absolute
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin


class RandomNoisePlugin(Plugin):
    """Dummy plugin for debugging.

    Example:
        >>> from synthcity.plugins import Plugins
        >>> plugin = Plugins().get("random_noise")
        >>> from sklearn.datasets import load_iris
        >>> X = load_iris()
        >>> plugin.fit(X)
        >>> plugin.generate()
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

    @staticmethod
    def name() -> str:
        return "random_noise"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "RandomNoisePlugin":
        return self

    def _generate(
        self, count: int, constraints: Constraints, **kwargs: Any
    ) -> pd.DataFrame:
        X_rnd = pd.DataFrame(
            np.zeros((count, len(self.schema().features()))),
            columns=self.schema().features(),
        )
        for feature in self.schema():
            sample = np.random.uniform(
                low=self.schema()[feature].min(),
                high=self.schema()[feature].max(),
                size=(count),
            )
            X_rnd[feature] = sample
        return X_rnd


plugin = RandomNoisePlugin
