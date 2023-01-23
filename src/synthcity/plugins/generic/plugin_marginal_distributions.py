# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class MarginalDistributionPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.generic.plugin_marginal_distributions.MarginalDistributionPlugin
        :parts: 1

    Synthetic data generation via marginal distributions.

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from synthcity.plugins import Plugins
        >>>
        >>> X, y = load_iris(as_frame = True, return_X_y = True)
        >>> X["target"] = y
        >>>
        >>> plugin = Plugins().get("marginal_distributions")
        >>> plugin.fit(X)
        >>>
        >>> plugin.generate(50)

    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            sampling_strategy="marginal",
        )

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
        self, X: DataLoader, *args: Any, **kwargs: Any
    ) -> "MarginalDistributionPlugin":
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _sample(count: int) -> pd.DataFrame:
            X_rnd = pd.DataFrame(
                np.zeros((count, len(self.training_schema().features()))),
                columns=self.training_schema().features(),
            )

            for feature in syn_schema:
                sample = syn_schema[feature].sample(count=count)
                X_rnd[feature] = sample
            return X_rnd

        return self._safe_generate(_sample, count, syn_schema)


plugin = MarginalDistributionPlugin
