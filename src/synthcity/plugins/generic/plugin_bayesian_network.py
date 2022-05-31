# stdlib
from typing import Any, List

# third party
import pandas as pd
import pgmpy.estimators as estimators
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import CategoricalDistribution, Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class BayesianNetworkPlugin(Plugin):
    """BayesianNetwork plugin.

    Example:
        >>> from synthcity.plugins import Plugins
        >>> plugin = Plugins().get("bayesian_network")
        >>> from sklearn.datasets import load_iris
        >>> X = load_iris()
        >>> plugin.fit(X)
        >>> plugin.generate()
    """

    def __init__(
        self,
        struct_learning_n_iter: int = 1000,
        struct_learning_search_method: str = "tree_search",  # hillclimb, pc, tree_search, mmhc, exhaustive
        struct_learning_score: str = "k2",  # k2, bdeu, bic, bds
        struct_max_indegree: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.struct_learning_n_iter = struct_learning_n_iter
        self.struct_learning_search_method = struct_learning_search_method
        self.struct_learning_score = struct_learning_score
        self.struct_max_indegree = struct_max_indegree

    @staticmethod
    def name() -> str:
        return "bayesian_network"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            CategoricalDistribution(
                name="struct_learning_search_method",
                choices=["hillclimb", "pc", "tree_search", "mmhc", "exhaustive"],
            ),
            CategoricalDistribution(
                name="struct_learning_score", choices=["k2", "bdeu", "bic", "bds"]
            ),
        ]

    def _get_structure_scorer(self) -> Any:
        return {
            "k2": estimators.K2Score,
            "bdeu": estimators.BDeuScore,
            "bic": estimators.BicScore,
            "bds": estimators.BDsScore,
        }[self.struct_learning_score]

    def _get_dag(self, X: pd.DataFrame) -> Any:
        scoring_method = scoring_method = self._get_structure_scorer()(data=X)
        if self.struct_learning_search_method == "hillclimb":
            return estimators.HillClimbSearch(data=X).estimate(
                scoring_method=scoring_method,
                max_indegree=self.struct_max_indegree,
                max_iter=self.struct_learning_n_iter,
                show_progress=False,
            )
        elif self.struct_learning_search_method == "pc":
            return estimators.PC(data=X).estimate(
                scoring_method=scoring_method, show_progress=False
            )
        elif self.struct_learning_search_method == "tree_search":
            return estimators.TreeSearch(data=X).estimate(show_progress=False)
        elif self.struct_learning_search_method == "mmhc":
            return estimators.MmhcEstimator(data=X).estimate(
                scoring_method=scoring_method,
            )
        elif self.struct_learning_search_method == "exhaustive":
            return estimators.ExhaustiveSearch(data=X).estimate()
        else:
            raise ValueError(f"invalid estimator {self.struct_learning_search_method}")

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "BayesianNetworkPlugin":
        df = X.dataframe()

        dag = self._get_dag(df)

        network = BayesianNetwork(dag)
        network.fit(df)

        self.model = BayesianModelSampling(network)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _sample(count: int) -> pd.DataFrame:
            vals = self.model.forward_sample(size=count, show_progress=False)

            return vals

        return self._safe_generate(_sample, count, syn_schema)


plugin = BayesianNetworkPlugin
