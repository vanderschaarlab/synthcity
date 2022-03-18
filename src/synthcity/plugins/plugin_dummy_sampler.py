# stdlib
from typing import Any, List, Optional

# third party
import pandas as pd

# synthcity absolute
from synthcity.plugins.core.constraints import Constraints
import synthcity.plugins.core.params as params
import synthcity.plugins.core.plugin as plugin


class DummySamplerPlugin(plugin.Plugin):
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
        return "resampling"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "DummySamplerPlugin":
        self.X = X
        return self

    def _generate(
        self,
        count: Optional[int] = None,
        constraints: Optional[Constraints] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        if self.X is None:
            raise RuntimeError("Fit the model first")

        if count is None:
            count = len(self.X)

        baseline = self.X

        if constraints:
            baseline = constraints.match(baseline)

        return baseline.sample(count, replace=True).reset_index(drop=True)


plugin = DummySamplerPlugin
