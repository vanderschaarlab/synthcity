# stdlib
from typing import Any, List

# third party
import pandas as pd
from sdv.tabular import CTGAN

# synthcity absolute
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class CTGANPlugin(Plugin):
    """CTGAN plugin.

    Example:
        >>> from synthcity.plugins import Plugins
        >>> plugin = Plugins().get("ctgan")
        >>> from sklearn.datasets import load_iris
        >>> X = load_iris()
        >>> plugin.fit(X)
        >>> plugin.generate()
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.model = CTGAN()

    @staticmethod
    def name() -> str:
        return "ctgan"

    @staticmethod
    def type() -> str:
        return "gan"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "CTGANPlugin":
        self.model.fit(X)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        constraints = syn_schema.as_constraints()

        data_synth = pd.DataFrame([], columns=self.schema().features())
        for it in range(self.sampling_patience):
            iter_samples = self.model.sample(count)
            iter_samples_df = pd.DataFrame(
                iter_samples, columns=self.schema().features()
            )
            iter_samples_df = syn_schema.adapt_dtypes(iter_samples_df)

            iter_synth_valid = constraints.match(iter_samples_df)
            data_synth = pd.concat([data_synth, iter_synth_valid], ignore_index=True)

            if len(data_synth) >= count:
                break

        data_synth = syn_schema.adapt_dtypes(data_synth).head(count)

        return data_synth


plugin = CTGANPlugin
