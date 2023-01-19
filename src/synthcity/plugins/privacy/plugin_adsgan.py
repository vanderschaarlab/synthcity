"""
Reference: Jinsung Yoon, Lydia N. Drumright, Mihaela van der Schaar,
        "Anonymization through Data Synthesis using Generative Adversarial Networks (ADS-GAN):
        A harmonizing advancement for AI in medicine,"
        IEEE Journal of Biomedical and Health Informatics (JBHI), 2019.
Paper link: https://ieeexplore.ieee.org/document/9034117
"""
# stdlib
from pathlib import Path
from typing import Any, List, Optional, Union

# third party
import pandas as pd

# Necessary packages
from pydantic import validate_arguments
from torch.utils.data import sampler

# synthcity absolute
from synthcity.metrics.weighted_metrics import WeightedMetrics
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.models.tabular_gan import TabularGAN
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE


class AdsGANPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.privacy.plugin_adsgan.AdsGANPlugin
        :parts: 1

    AdsGAN plugin - Anonymization through Data Synthesis using Generative Adversarial Networks.


    Args:
        generator_n_layers_hidden: int
            Number of hidden layers in the generator
        generator_n_units_hidden: int
            Number of hidden units in each layer of the Generator
        generator_nonlin: string, default 'leaky_relu'
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
        random_state: int
            random seed to use
        clipping_value: int, default 0
            Gradients clipping value. Zero disables the feature
        encoder_max_clusters: int
            The max number of clusters to create for continuous columns when encoding
        adjust_inference_sampling: bool
            Adjust the marginal probabilities in the synthetic data to closer match the training set. Active only with the ConditionalSampler
        # early stopping
        n_iter_print: int
            Number of iterations after which to print updates and check the validation loss.
        n_iter_min: int
            Minimum number of iterations to go through before starting early stopping
        patience: int
            Max number of iterations without any improvement before training early stopping is trigged.
        patience_metric: Optional[WeightedMetrics]
            If not None, the metric is used for evaluation the criterion for training early stopping.
        # Core Plugin arguments
        workspace: Path.
            Optional Path for caching intermediary results.
        compress_dataset: bool. Default = False.
            Drop redundant features before training the generator.
        sampling_patience: int.
            Max inference iterations to wait for the generated data to match the training schema.

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from synthcity.plugins import Plugins
        >>>
        >>> X, y = load_iris(as_frame = True, return_X_y = True)
        >>> X["target"] = y
        >>>
        >>> plugin = Plugins().get("adsgan", n_iter = 100)
        >>> plugin.fit(X)
        >>>
        >>> plugin.generate(50)
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_iter: int = 10000,
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
        # training
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 200,
        random_state: int = 0,
        clipping_value: int = 1,
        lambda_gradient_penalty: float = 10,
        lambda_identifiability_penalty: float = 0.1,
        encoder_max_clusters: int = 5,
        encoder: Any = None,
        dataloader_sampler: Optional[sampler.Sampler] = None,
        device: Any = DEVICE,
        adjust_inference_sampling: bool = False,
        # early stopping
        patience: int = 5,
        patience_metric: Optional[WeightedMetrics] = None,
        n_iter_print: int = 50,
        n_iter_min: int = 100,
        # core plugin arguments
        workspace: Path = Path("workspace"),
        compress_dataset: bool = False,
        sampling_patience: int = 500,
        **kwargs: Any
    ) -> None:
        super().__init__(
            device=device,
            random_state=random_state,
            sampling_patience=sampling_patience,
            workspace=workspace,
            compress_dataset=compress_dataset,
            **kwargs
        )
        if patience_metric is None:
            patience_metric = WeightedMetrics(
                metrics=[("detection", "detection_mlp")],
                weights=[1],
                workspace=workspace,
            )
        self.generator_n_layers_hidden = generator_n_layers_hidden
        self.generator_n_units_hidden = generator_n_units_hidden
        self.generator_nonlin = generator_nonlin
        self.n_iter = n_iter
        self.generator_dropout = generator_dropout
        self.generator_opt_betas = generator_opt_betas
        self.generator_extra_penalties = ["identifiability_penalty"]
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
        self.lambda_identifiability_penalty = lambda_identifiability_penalty

        self.encoder_max_clusters = encoder_max_clusters
        self.encoder = encoder
        self.dataloader_sampler = dataloader_sampler

        self.device = device
        self.patience = patience
        self.patience_metric = patience_metric
        self.n_iter_min = n_iter_min
        self.n_iter_print = n_iter_print
        self.adjust_inference_sampling = adjust_inference_sampling

    @staticmethod
    def name() -> str:
        return "adsgan"

    @staticmethod
    def type() -> str:
        return "privacy"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="generator_n_layers_hidden", low=1, high=4),
            IntegerDistribution(
                name="generator_n_units_hidden", low=50, high=150, step=50
            ),
            CategoricalDistribution(
                name="generator_nonlin", choices=["relu", "leaky_relu", "tanh", "elu"]
            ),
            FloatDistribution(name="generator_dropout", low=0, high=0.2),
            IntegerDistribution(name="discriminator_n_layers_hidden", low=1, high=4),
            IntegerDistribution(
                name="discriminator_n_units_hidden", low=50, high=150, step=50
            ),
            CategoricalDistribution(
                name="discriminator_nonlin",
                choices=["relu", "leaky_relu", "tanh", "elu"],
            ),
            FloatDistribution(name="discriminator_dropout", low=0, high=0.2),
            CategoricalDistribution(name="lr", choices=[1e-3, 2e-4, 1e-4]),
            CategoricalDistribution(name="weight_decay", choices=[1e-3, 1e-4]),
            IntegerDistribution(name="encoder_max_clusters", low=2, high=20),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "AdsGANPlugin":
        cond: Optional[Union[pd.DataFrame, pd.Series]] = None
        if "cond" in kwargs:
            cond = kwargs["cond"]

        self.model = TabularGAN(
            X.dataframe(),
            cond=cond,
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
            generator_extra_penalties=self.generator_extra_penalties,
            discriminator_n_units_hidden=self.discriminator_n_units_hidden,
            discriminator_n_layers_hidden=self.discriminator_n_layers_hidden,
            discriminator_n_iter=self.discriminator_n_iter,
            discriminator_nonlin=self.discriminator_nonlin,
            discriminator_batch_norm=False,
            discriminator_dropout=self.discriminator_dropout,
            discriminator_lr=self.lr,
            discriminator_weight_decay=self.weight_decay,
            discriminator_opt_betas=self.discriminator_opt_betas,
            encoder=self.encoder,
            clipping_value=self.clipping_value,
            lambda_gradient_penalty=self.lambda_gradient_penalty,
            lambda_identifiability_penalty=self.lambda_identifiability_penalty,
            encoder_max_clusters=self.encoder_max_clusters,
            dataloader_sampler=self.dataloader_sampler,
            device=self.device,
            patience=self.patience,
            patience_metric=self.patience_metric,
            n_iter_min=self.n_iter_min,
            n_iter_print=self.n_iter_print,
            adjust_inference_sampling=self.adjust_inference_sampling,
        )
        self.model.fit(X.dataframe(), cond=cond)

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> DataLoader:
        cond: Optional[Union[pd.DataFrame, pd.Series]] = None
        if "cond" in kwargs:
            cond = kwargs["cond"]

        return self._safe_generate(self.model.generate, count, syn_schema, cond=cond)


plugin = AdsGANPlugin
