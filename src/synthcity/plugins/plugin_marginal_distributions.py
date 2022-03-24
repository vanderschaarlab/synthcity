# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd

# synthcity absolute
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class MarginalDistributionPlugin(Plugin):
    """Synthetic data generation via marginal distributions.

    Example:
        >>> from synthcity.plugins import Plugins
        >>> plugin = Plugins().get("marginal_distributions")
        >>> from sklearn.datasets import load_iris
        >>> X = load_iris()
        >>> plugin.fit(X)
        >>> plugin.generate()
    """

    def __init__(self, epsilon: float = 1.0, **kwargs: Any) -> None:
        """
        Args:
            epsilon: float
                Privacy parameter epsilon in differential privacy. >= 0.
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon

    @staticmethod
    def name() -> str:
        return "marginal_distributions"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return []

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "MarginalDistributionPlugin":
        self.local_epsilon = self.epsilon / X.shape[1]

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


plugin = MarginalDistributionPlugin
