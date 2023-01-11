# stdlib
from typing import Any, List

# third party
import pandas as pd

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class DummySamplerPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.generic.plugin_dummy_sampler.DummySamplerPlugin
        :parts: 1

    Dummy sampling plugin for debugging.

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from synthcity.plugins import Plugins
        >>>
        >>> X, y = load_iris(as_frame = True, return_X_y = True)
        >>> X["target"] = y
        >>>
        >>> plugin = Plugins().get("dummy_sampler")
        >>> plugin.fit(X)
        >>>
        >>> plugin.generate(50)


    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "dummy_sampler"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "DummySamplerPlugin":
        self.X = X.dataframe()
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _sample(count: int) -> pd.DataFrame:
            baseline = self.X
            constraints = syn_schema.as_constraints()

            baseline = constraints.match(baseline)
            if len(baseline) == 0:
                raise ValueError("Cannot generate data")

            if len(baseline) <= count:
                return baseline.sample(frac=1)

            return baseline.sample(count, replace=True).reset_index(drop=True)

        return self._safe_generate(_sample, count, syn_schema)


plugin = DummySamplerPlugin
