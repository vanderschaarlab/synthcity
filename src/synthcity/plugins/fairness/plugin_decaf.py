"""DECAF (DEbiasing CAusal Fairness)

Reference: Boris van Breugel, Trent Kyono, Jeroen Berrevoets, Mihaela van der Schaar
"DECAF: Generating Fair Synthetic Data Using Causally-Aware Generative Networks"(2021).
"""

# stdlib
from typing import Any, List, Tuple

# third party
import pandas as pd
import pgmpy.estimators as estimators
import pytorch_lightning as pl
import torch
from decaf import DECAF, DataModule

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import CategoricalDistribution, Distribution
from synthcity.plugins.core.models.tabular_encoder import TabularEncoder
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema

accelerator = "gpu" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DECAFPlugin(Plugin):
    """DECAF plugin.

    Example:
        >>> from synthcity.plugins import Plugins
        >>> plugin = Plugins().get("decaf")
        >>> from sklearn.datasets import load_iris
        >>> X = load_iris()
        >>> plugin.fit(X)
        >>> plugin.generate()
    """

    def __init__(
        self,
        n_iter: int = 1000,
        n_units_hidden: int = 500,
        lr: float = 1e-3,
        batch_size: int = 200,
        lambda_gp: float = 10,
        lambda_privacy: float = 1,
        eps: float = 1e-8,
        alpha: float = 1,
        rho: float = 1,
        weight_decay: float = 1e-2,
        grad_dag_loss: bool = False,
        l1_g: float = 0,
        l1_W: float = 1,
        struct_learning_enabled: bool = False,
        struct_learning_n_iter: int = 1000,
        struct_learning_search_method: str = "tree_search",  # hillclimb, pc, tree_search, mmhc, exhaustive
        struct_learning_score: str = "k2",  # k2, bdeu, bic, bds
        struct_max_indegree: int = 4,
        encoder_max_clusters: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.n_iter = n_iter
        self.n_units_hidden = n_units_hidden
        self.lr = lr
        self.batch_size = batch_size
        self.lambda_gp = lambda_gp
        self.lambda_privacy = lambda_privacy
        self.eps = eps
        self.alpha = alpha
        self.rho = rho
        self.weight_decay = weight_decay
        self.grad_dag_loss = grad_dag_loss
        self.l1_g = l1_g
        self.l1_W = l1_W

        self.struct_learning_enabled = struct_learning_enabled
        self.struct_learning_n_iter = struct_learning_n_iter
        self.struct_learning_search_method = struct_learning_search_method
        self.struct_learning_score = struct_learning_score
        self.struct_max_indegree = struct_max_indegree

        self.encoder = TabularEncoder(max_clusters=encoder_max_clusters)

    @staticmethod
    def name() -> str:
        return "decaf"

    @staticmethod
    def type() -> str:
        return "fairness"

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

    def _fit(
        self, X: DataLoader, *args: Any, dag: List[Tuple[int, int]] = [], **kwargs: Any
    ) -> "DECAFPlugin":
        df = X.dataframe()
        df = self.encoder.fit_transform(df)

        if dag == [] and self.struct_learning_enabled:
            dag = self._get_dag(df).edges()

        dm = DataModule(df)
        self.features = X.columns
        self.encoded_features = df.columns

        # TODO: find workaround for data
        self.raw_data = dm.dataset.x

        self.model = DECAF(
            dm.dims[0],
            dag_seed=dag,
            h_dim=self.n_units_hidden,
            lr=self.lr,
            batch_size=self.batch_size,
            lambda_gp=self.lambda_gp,
            lambda_privacy=self.lambda_privacy,
            eps=self.eps,
            alpha=self.alpha,
            rho=self.rho,
            weight_decay=self.weight_decay,
            grad_dag_loss=self.grad_dag_loss,
            l1_g=self.l1_g,
            l1_W=self.l1_W,
            nonlin_out=self.encoder.activation_layout(
                discrete_activation="softmax",
                continuous_activation="none",
            ),
        ).to(DEVICE)
        trainer = pl.Trainer(
            accelerator=accelerator, max_epochs=self.n_iter, logger=False
        )
        trainer.fit(self.model, dm)

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _sample(count: int) -> pd.DataFrame:
            vals = (
                self.model.gen_synthetic(self.raw_data, **kwargs).detach().cpu().numpy()
            )

            output = self.encoder.inverse_transform(
                pd.DataFrame(vals, columns=self.encoded_features)
            ).sample(count)

            return output

        return self._safe_generate(_sample, count, syn_schema)


plugin = DECAFPlugin
