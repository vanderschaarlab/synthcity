"""PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees Codebase.

Reference: James Jordon, Jinsung Yoon, Mihaela van der Schaar,
"PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees,"
International Conference on Learning Representations (ICLR), 2019.
Paper link: https://openreview.net/forum?id=S1zk9iRqF7
"""
# stdlib
from typing import Any, List

# third party
import pandas as pd

# Necessary packages
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.generic.extras_pategan import PATEGAN


class PATEGANPlugin(Plugin):
    """PATEGAN plugin.

    Args:
        generator_n_layers_hidden: int
            Number of hidden layers in the generator
        generator_n_units_hidden: int
            Number of hidden units in each layer of the Generator
        generator_nonlin: string, default 'tanh'
            Nonlinearity to use in the generator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        n_iter: int
            Maximum number of iterations in the Generator.
        generator_dropout: float
            Dropout value. If 0, the dropout is not used.
        discriminator_n_layers_hidden: int
            Number of hidden layers in the discriminator
        discriminator_n_units_hidden: int
            Number of hidden units in each layer of the discriminator
        discriminator_nonlin: string, default 'leaky_relu'
            Nonlinearity to use in the discriminator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        discriminator_n_iter: int
            Maximum number of iterations in the discriminator.
        discriminator_dropout: float
            Dropout value for the discriminator. If 0, the dropout is not used.
        lr: float
            learning rate for optimizer. step_size equivalent in the JAX version.
        weight_decay: float
            l2 (ridge) penalty for the weights.
        batch_size: int
            Batch size
        seed: int
            Seed used
        clipping_value: int, default 0
            Gradients clipping value. Zero disables the feature
        n_teachers: int
            Number of teachers to train
        teacher_template: str
            Model to use for the teachers. Can be linear, xgboost.
        epsilon: float
            Differential privacy parameter
        delta: float
            Differential privacy parameter
        lambda: float
            Noise size
        encoder_max_clusters: int
            The max number of clusters to create for continuous columns when encoding


    Example:
        >>> from synthcity.plugins import Plugins
        >>> plugin = Plugins().get("pategan")
        >>> from sklearn.datasets import load_iris
        >>> X = load_iris()
        >>> plugin.fit(X)
        >>> plugin.generate()
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        # GAN
        n_iter: int = 10,
        generator_n_iter: int = 10,
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 100,
        generator_nonlin: str = "tanh",
        generator_dropout: float = 0,
        discriminator_n_layers_hidden: int = 2,
        discriminator_n_units_hidden: int = 100,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        batch_size: int = 500,
        seed: int = 0,
        clipping_value: int = 0,
        encoder_max_clusters: int = 20,
        # Privacy
        n_teachers: int = 10,
        teacher_template: str = "xgboost",
        epsilon: float = 10.0,
        delta: float = 0.00001,
        lamda: float = 1,
        alpha: int = 20,
        encoder: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.model = PATEGAN(
            max_iter=n_iter,
            generator_n_layers_hidden=generator_n_layers_hidden,
            generator_n_units_hidden=generator_n_units_hidden,
            generator_nonlin=generator_nonlin,
            generator_n_iter=generator_n_iter,
            generator_dropout=generator_dropout,
            discriminator_n_layers_hidden=discriminator_n_layers_hidden,
            discriminator_n_units_hidden=discriminator_n_units_hidden,
            discriminator_nonlin=discriminator_nonlin,
            discriminator_n_iter=discriminator_n_iter,
            discriminator_dropout=discriminator_dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            seed=seed,
            clipping_value=clipping_value,
            encoder_max_clusters=encoder_max_clusters,
            encoder=encoder,
            # Privacy
            n_teachers=n_teachers,
            teacher_template=teacher_template,
            epsilon=epsilon,
            delta=delta,
            lamda=lamda,
        )

    @staticmethod
    def name() -> str:
        return "pategan"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="n_iter", low=1, high=15),
            IntegerDistribution(name="generator_n_layers_hidden", low=1, high=4),
            IntegerDistribution(
                name="generator_n_units_hidden", low=50, high=150, step=50
            ),
            CategoricalDistribution(
                name="generator_nonlin", choices=["relu", "leaky_relu", "tanh", "elu"]
            ),
            IntegerDistribution(name="generator_n_iter", low=1, high=10),
            FloatDistribution(name="generator_dropout", low=0, high=0.2),
            IntegerDistribution(name="discriminator_n_layers_hidden", low=1, high=4),
            IntegerDistribution(
                name="discriminator_n_units_hidden", low=50, high=150, step=50
            ),
            CategoricalDistribution(
                name="discriminator_nonlin",
                choices=["relu", "leaky_relu", "tanh", "elu"],
            ),
            IntegerDistribution(name="discriminator_n_iter", low=1, high=5),
            FloatDistribution(name="discriminator_dropout", low=0, high=0.2),
            CategoricalDistribution(name="lr", choices=[1e-3, 2e-4, 1e-4]),
            CategoricalDistribution(name="weight_decay", choices=[1e-3, 1e-4]),
            CategoricalDistribution(name="batch_size", choices=[64, 128, 256, 512]),
            IntegerDistribution(name="n_teachers", low=5, high=15),
            CategoricalDistribution(
                name="teacher_template", choices=["linear", "xgboost"]
            ),
            IntegerDistribution(name="encoder_max_clusters", low=2, high=20),
            FloatDistribution(name="lamda", low=1, high=10),
            CategoricalDistribution(
                name="delta", choices=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
            ),
            IntegerDistribution(name="alpha", low=2, high=50),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "PATEGANPlugin":
        self.model.fit(X.dataframe())

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.sample, count, syn_schema)


plugin = PATEGANPlugin
