"""
Reference: Boris van Breugel, Trent Kyono, Jeroen Berrevoets, Mihaela van der Schaar "DECAF: Generating Fair Synthetic Data Using Causally-Aware Generative Networks"(2021).
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
import synthcity.logger as log
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import CategoricalDistribution, Distribution
from synthcity.plugins.core.models.dag.dstruct import get_dstruct_dag
from synthcity.plugins.core.models.tabular_gan import TabularGAN
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema

accelerator = "gpu" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DECAFPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.privacy.plugin_decaf.DECAFPlugin
        :parts: 1

    DECAF (DEbiasing CAusal Fairness) plugin.

    Args:
        n_iter: int
            Number of training iterations.
        generator_n_layers_hidden: int
            Number of hidden layers in the generator.
        generator_n_units_hidden
            Number of neurons in the hidden layers of the generator.
        generator_nonlin: str
            Nonlinearity used by the generator for the hidden layers: leaky_relu, relu, gelu etc.
        generator_dropout: float
            Generator dropout.
        generator_opt_betas: tuple
            Generator  initial decay rates for the Adam optimizer
        discriminator_n_layers_hidden: int
            Number of hidden layers in the discriminator.
        discriminator_n_units_hidden: int
            Number of neurons in the hidden layers of the discriminator.
        discriminator_nonlin: str
            Nonlinearity used by the discriminator for the hidden layers: leaky_relu, relu, gelu etc.
        discriminator_n_iter: int
            Discriminator number of iterations(default = 1)
        discriminator_dropout: float
            Discriminator dropout
        discriminator_opt_betas: tuple
            Discriminator  initial decay rates for the Adam optimizer
        lr: float
            Learning rate
        weight_decay: float
            Optimizer weight decay
        batch_size: int
            Batch size
        random_state: int
            Random seed
        clipping_value: int
            Gradient clipping value
        lambda_gradient_penalty: float
            Gradient penalty factor used for training the GAN.
        lambda_privacy: float
            Privacy factor used the AdsGAN loss.
        eps: float = 1e-8,
            Noise added to the privacy loss
        alpha: float
            Gradient penalty weight for real samples.
        rho: float
            DAG loss factor
        l1_g: float = 0
            l1 regularization loss for the generator
        l1_W: float = 1
            l1 regularization factor for l1_g
        struct_learning_enabled: bool
            Enable DAG learning outside DECAF.
        struct_learning_n_iter: int
            Number of iterations for the DAG search.
        struct_learning_search_method: str
            DAG search strategy: hillclimb, pc, tree_search, mmhc, exhaustive, d-struct
        struct_learning_score: str
            DAG search scoring strategy: k2, bdeu, bic, bds
        struct_max_indegree: int
            Max parents in the DAG.
        encoder_max_clusters: int
            Number of clusters used for tabular encoding
        device: Any = DEVICE
            torch device used for training.


    Example:
        >>> from sklearn.datasets import load_iris
        >>> from synthcity.plugins import Plugins
        >>>
        >>> X, y = load_iris(as_frame = True, return_X_y = True)
        >>> X["target"] = y
        >>>
        >>> plugin = Plugins().get("decaf", n_iter = 100)
        >>> plugin.fit(X)
        >>>
        >>> plugin.generate(50)
    """

    def __init__(
        self,
        n_iter: int = 1000,
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 500,
        generator_nonlin: str = "relu",
        generator_dropout: float = 0.1,
        generator_opt_betas: tuple = (0.5, 0.999),
        discriminator_n_layers_hidden: int = 2,
        discriminator_n_units_hidden: int = 500,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_dropout: float = 0.1,
        discriminator_opt_betas: tuple = (0.5, 0.999),
        lr: float = 1e-3,
        batch_size: int = 500,
        random_state: int = 0,
        clipping_value: int = 1,
        lambda_gradient_penalty: float = 10,
        lambda_privacy: float = 1,
        eps: float = 1e-8,
        alpha: float = 1,
        rho: float = 1,
        weight_decay: float = 1e-2,
        l1_g: float = 0,
        l1_W: float = 1,
        grad_dag_loss: bool = False,
        struct_learning_enabled: bool = True,
        struct_learning_n_iter: int = 1000,
        struct_learning_search_method: str = "tree_search",  # hillclimb, pc, tree_search, mmhc, exhaustive, d-struct
        struct_learning_score: str = "k2",  # k2, bdeu, bic, bds
        struct_max_indegree: int = 4,
        encoder_max_clusters: int = 10,
        device: Any = DEVICE,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.n_iter = n_iter
        self.generator_n_layers_hidden = generator_n_layers_hidden
        self.generator_n_units_hidden = generator_n_units_hidden
        self.generator_nonlin = generator_nonlin
        self.generator_dropout = generator_dropout
        self.generator_opt_betas = generator_opt_betas
        self.discriminator_n_layers_hidden = discriminator_n_layers_hidden
        self.discriminator_n_units_hidden = discriminator_n_units_hidden
        self.discriminator_nonlin = discriminator_nonlin
        self.discriminator_n_iter = discriminator_n_iter
        self.discriminator_dropout = discriminator_dropout
        self.discriminator_opt_betas = discriminator_opt_betas

        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.random_state = random_state
        self.clipping_value = clipping_value
        self.lambda_gradient_penalty = lambda_gradient_penalty
        self.lambda_privacy = lambda_privacy
        self.grad_dag_loss = grad_dag_loss

        self.device = device

        self.eps = eps
        self.alpha = alpha
        self.rho = rho
        self.weight_decay = weight_decay
        self.l1_g = l1_g
        self.l1_W = l1_W

        self.struct_learning_enabled = struct_learning_enabled
        self.struct_learning_n_iter = struct_learning_n_iter
        self.struct_learning_search_method = struct_learning_search_method
        self.struct_learning_score = struct_learning_score
        self.struct_max_indegree = struct_max_indegree

        self.encoder_max_clusters = encoder_max_clusters

    @staticmethod
    def name() -> str:
        return "decaf"

    @staticmethod
    def type() -> str:
        return "privacy"

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
        if self.struct_learning_search_method == "d-struct":
            return get_dstruct_dag(
                X,
                batch_size=self.batch_size,
                seed=self.random_state,
                n_iter=self.n_iter,
            )

        scoring_method = scoring_method = self._get_structure_scorer()(data=X)
        if self.struct_learning_search_method == "hillclimb":
            raw_dag = estimators.HillClimbSearch(data=X).estimate(
                scoring_method=scoring_method,
                max_indegree=self.struct_max_indegree,
                max_iter=self.struct_learning_n_iter,
                show_progress=False,
            )
        elif self.struct_learning_search_method == "pc":
            raw_dag = estimators.PC(data=X).estimate(
                scoring_method=scoring_method, show_progress=False
            )
        elif self.struct_learning_search_method == "tree_search":
            raw_dag = estimators.TreeSearch(data=X).estimate(show_progress=False)
        elif self.struct_learning_search_method == "mmhc":
            raw_dag = estimators.MmhcEstimator(data=X).estimate(
                scoring_method=scoring_method,
            )
        elif self.struct_learning_search_method == "exhaustive":
            raw_dag = estimators.ExhaustiveSearch(data=X).estimate()
        else:
            raise ValueError(f"invalid estimator {self.struct_learning_search_method}")

        raw_dag = raw_dag.edges()
        dag = []
        for src, dst in raw_dag:
            dag.append(
                (
                    X.columns.values.tolist().index(src),
                    X.columns.values.tolist().index(dst),
                )
            )

        return dag

    def _fit(
        self, X: DataLoader, *args: Any, dag: List[Tuple[int, int]] = [], **kwargs: Any
    ) -> "DECAFPlugin":
        # train the baseline generator
        log.info("[DECAF] train baseline generator")
        self.baseline_generator = TabularGAN(
            X.dataframe(),
            n_units_latent=self.generator_n_units_hidden,
            batch_size=self.batch_size,
            generator_n_layers_hidden=self.generator_n_layers_hidden,
            generator_n_units_hidden=self.generator_n_units_hidden,
            generator_nonlin=self.generator_nonlin,
            generator_nonlin_out_discrete="softmax",
            generator_nonlin_out_continuous="none",
            generator_lr=self.lr,
            generator_residual=True,
            generator_n_iter=self.n_iter,
            generator_batch_norm=False,
            generator_dropout=0,
            generator_weight_decay=self.weight_decay,
            generator_opt_betas=self.generator_opt_betas,
            generator_extra_penalties=[],
            discriminator_n_units_hidden=self.discriminator_n_units_hidden,
            discriminator_n_layers_hidden=self.discriminator_n_layers_hidden,
            discriminator_n_iter=self.discriminator_n_iter,
            discriminator_nonlin=self.discriminator_nonlin,
            discriminator_batch_norm=False,
            discriminator_dropout=self.discriminator_dropout,
            discriminator_lr=self.lr,
            discriminator_weight_decay=self.weight_decay,
            discriminator_opt_betas=self.discriminator_opt_betas,
            clipping_value=self.clipping_value,
            lambda_gradient_penalty=self.lambda_gradient_penalty,
            encoder_max_clusters=self.encoder_max_clusters,
            device=self.device,
        )
        self.baseline_generator.fit(X.dataframe())

        # train the debiasing generator
        df = X.dataframe()
        df = self.baseline_generator.encode(df)

        if dag == [] and self.struct_learning_enabled:
            dag = self._get_dag(df)

        log.info(f"[DECAF] using DAG {dag}")

        dm = DataModule(df)
        self.features = X.columns
        self.encoded_features = df.columns

        log.info("[DECAF] train debiasing generator")
        self.model = DECAF(
            dm.dims[0],
            dag_seed=dag,
            h_dim=self.generator_n_units_hidden,
            lr=self.lr,
            batch_size=self.batch_size,
            lambda_gp=self.lambda_gradient_penalty,
            lambda_privacy=self.lambda_privacy,
            eps=self.eps,
            alpha=self.alpha,
            rho=self.rho,
            weight_decay=self.weight_decay,
            grad_dag_loss=self.grad_dag_loss,
            l1_g=self.l1_g,
            l1_W=self.l1_W,
            nonlin_out=self.baseline_generator.encoder.activation_layout(
                discrete_activation="softmax",
                continuous_activation="none",
            ),
        ).to(DEVICE)
        trainer = pl.Trainer(
            accelerator=accelerator,
            max_epochs=self.n_iter,
            logger=False,
        )
        trainer.fit(self.model, dm)

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _sample(count: int) -> pd.DataFrame:
            # generate baseline values
            seed_values = self.baseline_generator(count)
            seed_values = torch.from_numpy(seed_values).to(DEVICE)
            # debias baseline values
            vals = (
                self.model.gen_synthetic(seed_values, **kwargs).detach().cpu().numpy()
            )

            output = self.baseline_generator.decode(
                pd.DataFrame(vals, columns=self.encoded_features)
            ).sample(count)

            return output

        return self._safe_generate(_sample, count, syn_schema)


plugin = DECAFPlugin
