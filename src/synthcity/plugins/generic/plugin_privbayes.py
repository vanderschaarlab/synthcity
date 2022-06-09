"""
Adapted from:
    - https://github.com/daanknoors/synthetic_data_generation
    - https://github.com/DataResponsibly/DataSynthesizer
"""
# stdlib
from typing import Any, List

# third party
import pandas as pd

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.generic.extras_privbayes import PrivBayes


class PrivBayesPlugin(Plugin):
    """PrivBayes algorithm.

    Paper: PrivBayes: Private Data Release via Bayesian Networks. (2017), Zhang J, Cormode G, Procopiuc CM, Srivastava D, Xiao X."""

    def __init__(
        self,
        dp_epsilon: float = 1.0,
        theta_usefulness: float = 4,
        epsilon_split: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.dp_epsilon = dp_epsilon
        self.theta_usefulness = theta_usefulness
        self.epsilon_split = epsilon_split

    @staticmethod
    def name() -> str:
        return "privbayes"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "PrivBayesPlugin":
        self.model = PrivBayes(
            epsilon=self.dp_epsilon,
            theta_usefulness=self.theta_usefulness,
            epsilon_split=self.epsilon_split,
            score_function="R",
        )
        self.model.fit(X.dataframe())
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.sample, count, syn_schema)


plugin = PrivBayesPlugin
