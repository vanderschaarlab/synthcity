# stdlib
from typing import Any, List, Optional

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
        >>> plugin = Plugins().get("dummy_sampler")
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
        return "random"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "RandomNoisePlugin":
        self.features = list(X.columns)
        self.length = len(X)
        return self

    def _generate(
        self,
        count: Optional[int] = None,
        constraints: Optional[Constraints] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        if self.features is None:
            raise RuntimeError("Fit the model first")
        if constraints is None:
            constraints = self.schema().as_constraint()

        if count is None:
            count = self.length

        X_rnd = pd.DataFrame(
            np.zeros((count, len(self.features))), columns=self.features
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
