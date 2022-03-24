# stdlib
from typing import Any, List

# third party
import pandas as pd

# synthcity absolute
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class DummySamplerPlugin(Plugin):
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
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "dummy_sampler"

    @staticmethod
    def type() -> str:
        return "sampling"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "DummySamplerPlugin":
        self.X = X
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        if self.X is None:
            raise RuntimeError("Fit the model first")

        baseline = self.X
        constraints = syn_schema.as_constraints()

        baseline = constraints.match(baseline)

        return baseline.sample(count, replace=True).reset_index(drop=True)


plugin = DummySamplerPlugin
