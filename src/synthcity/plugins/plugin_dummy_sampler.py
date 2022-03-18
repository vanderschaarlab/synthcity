# stdlib
from typing import Any, List

# third party
import pandas as pd

# synthcity absolute
import synthcity.plugins.core.base as base
import synthcity.plugins.core.params as params


class DummySamplerPlugin(base.RegressionPlugin):
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
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "DummySamplerPlugin":
        self.X = X
        return self

    def _generate(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.X.sample(10)


plugin = DummySamplerPlugin
