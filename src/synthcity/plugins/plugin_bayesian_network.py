# stdlib
from typing import Any, List

# third party
import pandas as pd
from pomegranate import BayesianNetwork

# synthcity absolute
from synthcity.plugins.core.distribution import CategoricalDistribution, Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class BayesianNetworkPlugin(Plugin):
    """Standard Bayesian network"""

    def __init__(
        self,
        training_algorithm: str = "greedy",  # greedy, chow-liu or exact
        sampling_algorithm: str = "gibbs",  # gibbs or rejection
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.sampling_algorithm = sampling_algorithm
        self.training_algorithm = training_algorithm

    @staticmethod
    def name() -> str:
        return "bayesian_network"

    @staticmethod
    def type() -> str:
        return "bayesian"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            CategoricalDistribution(
                name="training_algorithm", choices=["greedy", "chow-liu", "exact"]
            ),
            CategoricalDistribution(
                name="sampling_algorithm", choices=["gibbs", "rejection"]
            ),
        ]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "BayesianNetworkPlugin":
        self.model = BayesianNetwork.from_samples(
            X.to_numpy(), algorithm=self.training_algorithm
        )
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        constraints = syn_schema.as_constraints()

        data_synth = pd.DataFrame([], columns=self.schema().features())
        for it in range(100):
            iter_samples = self.model.sample(n=count, algorithm=self.sampling_algorithm)
            iter_samples_df = pd.DataFrame(
                iter_samples, columns=self.schema().features()
            )

            iter_synth_valid = constraints.match(iter_samples_df)
            data_synth = pd.concat([data_synth, iter_synth_valid], ignore_index=True)

            if len(data_synth) >= count:
                break

        data_synth = syn_schema.adapt_dtypes(data_synth).head(count)

        return data_synth


plugin = BayesianNetworkPlugin
