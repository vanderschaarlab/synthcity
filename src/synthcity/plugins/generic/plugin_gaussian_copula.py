# stdlib
from typing import Any, List

# third party
import pandas as pd
from sdv.tabular import GaussianCopula

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class GaussianCopulaPlugin(Plugin):
    """GaussianCopula plugin.

    Example:
        >>> from synthcity.plugins import Plugins
        >>> plugin = Plugins().get("gaussian_copula")
        >>> from sklearn.datasets import load_iris
        >>> X = load_iris()
        >>> plugin.fit(X)
        >>> plugin.generate()
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.model = GaussianCopula()

    @staticmethod
    def name() -> str:
        return "gaussian_copula"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "GaussianCopulaPlugin":
        self.model.fit(X.dataframe())
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.sample, count, syn_schema)


plugin = GaussianCopulaPlugin
