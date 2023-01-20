"""
Reference: Jankan, Ankur and Panda, Abinash, "pgmpy: Probabilistic graphical models using python,"
        Proceedings of the 14th Python in Science Conference (SCIPY 2015), 2015.
"""

# stdlib
from pathlib import Path
from typing import Any, List

# third party
import numpy as np
import pandas as pd
import pgmpy.estimators as estimators
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import CategoricalDistribution, Distribution
from synthcity.plugins.core.models.tabular_encoder import TabularEncoder
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class BayesianNetworkPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.generic.plugin_bayesian_network.BayesianNetworkPlugin
        :parts: 1

    Bayesian Network for generative modeling. Implemented using pgmpy backend.
    Args:
        struct_learning_n_iter: int
            Number of iterations for the DAG learning
        struct_learning_search_method: str = "tree_search"
             Search method for learning the DAG: hillclimb, pc, tree_search, mmhc, exhaustive
        struct_learning_score: str = "k2",
            Scoring for the DAG search: k2, bdeu, bic, bds
        struct_max_indegree: int = 4
            The maximum number of parents for each node.
        encoder_max_clusters: int = 10
            Data encoding clusters.
        encoder_noise_scale: float.
            Small noise to add to the final data, to prevent data leakage.
        workspace: Path.
            Optional Path for caching intermediary results.
        compress_dataset: bool. Default = False.
            Drop redundant features before training the generator.
        random_state: int.
            Random seed.
        sampling_patience: int.
            Max inference iterations to wait for the generated data to match the training schema.

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from synthcity.plugins import Plugins
        >>>
        >>> X, y = load_iris(as_frame = True, return_X_y = True)
        >>> X["target"] = y
        >>>
        >>> plugin = Plugins().get("bayesian_network")
        >>> plugin.fit(X)
        >>>
        >>> plugin.generate(50)

    """

    def __init__(
        self,
        struct_learning_n_iter: int = 1000,
        struct_learning_search_method: str = "tree_search",  # hillclimb, pc, tree_search, mmhc, exhaustive
        struct_learning_score: str = "k2",  # k2, bdeu, bic, bds
        struct_max_indegree: int = 4,
        encoder_max_clusters: int = 10,
        encoder_noise_scale: float = 0.1,
        # core plugin
        workspace: Path = Path("workspace"),
        compress_dataset: bool = False,
        random_state: int = 0,
        sampling_patience: int = 500,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            random_state=random_state,
            sampling_patience=sampling_patience,
            workspace=workspace,
            compress_dataset=compress_dataset,
            **kwargs,
        )

        self.struct_learning_n_iter = struct_learning_n_iter
        self.struct_learning_search_method = struct_learning_search_method
        self.struct_learning_score = struct_learning_score
        self.struct_max_indegree = struct_max_indegree

        self.encoder = TabularEncoder(max_clusters=encoder_max_clusters)
        self.encoder_noise_scale = encoder_noise_scale

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
                choices=["hillclimb", "pc", "tree_search"],
            ),
            CategoricalDistribution(
                name="struct_learning_score", choices=["k2", "bdeu", "bic", "bds"]
            ),
        ]

    def _encode_decode(self, data: pd.DataFrame) -> pd.DataFrame:
        encoded = self.encoder.transform(data)

        # add noise to the mixture means, but keep the continuous cluster
        noise = np.random.normal(
            loc=0, scale=self.encoder_noise_scale, size=len(encoded)
        )
        for col in encoded.columns:
            if col.endswith(".value"):
                encoded[col] += noise

        decoded = self.encoder.inverse_transform(encoded)
        decoded = decoded[data.columns]

        return decoded

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
        self.encoder.fit(df)

        dag = self._get_dag(df)

        network = BayesianNetwork(dag)
        network.fit(df)

        self.model = BayesianModelSampling(network)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _sample(count: int) -> pd.DataFrame:
            vals = self.model.forward_sample(size=count, show_progress=False)

            return self._encode_decode(vals)

        return self._safe_generate(_sample, count, syn_schema)


plugin = BayesianNetworkPlugin
