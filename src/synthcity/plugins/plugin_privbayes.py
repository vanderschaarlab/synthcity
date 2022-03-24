# stdlib
from typing import Any, List

# third party
import pandas as pd
from synthesis.synthesizers import PrivBayes

# synthcity absolute
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class PrivBayesPlugin(Plugin):
    """PrivBayes algorithm.

    Paper: PrivBayes: Private Data Release via Bayesian Networks. (2017), Zhang J, Cormode G, Procopiuc CM, Srivastava D, Xiao X."""

    def __init__(
        self,
        dp_epsilon: float = 1.0,
        theta_usefulness: float = 4,
        epsilon_split: float = 0.3,
        score_function: str = "R",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.theta_usefulness = theta_usefulness
        self.epsilon_split = epsilon_split
        self.score_function = score_function

    @staticmethod
    def name() -> str:
        return "privbayes"

    @staticmethod
    def type() -> str:
        return "bayesian"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "PrivBayesPlugin":
        self.model = PrivBayes(
            epsilon=self.dp_epsilon,
            theta_usefulness=self.theta_usefulness,
            epsilon_split=self.epsilon_split,
            score_function=self.score_function,
            verbose=False,
        )
        self.model.fit(X)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        constraints = syn_schema.as_constraints()

        data_synth = pd.DataFrame([], columns=self.schema().features())
        for it in range(100):
            iter_samples = self.model.sample(count)
            iter_samples_df = pd.DataFrame(
                iter_samples, columns=self.schema().features()
            )

            iter_synth_valid = constraints.match(iter_samples_df)
            data_synth = pd.concat([data_synth, iter_synth_valid], ignore_index=True)

            if len(data_synth) >= count:
                break

        data_synth = syn_schema.adapt_dtypes(data_synth).head(count)

        return data_synth


plugin = PrivBayesPlugin
