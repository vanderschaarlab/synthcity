# stdlib
from pathlib import Path
from typing import Any, List, Optional

# third party
# Necessary packages
from pydantic import validate_arguments

# synthcity absolute
from synthcity.metrics.weighted_metrics import WeightedMetrics
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
)
from synthcity.plugins.core.models.convnet import (
    suggest_image_generator_discriminator_arch,
)
from synthcity.plugins.core.models.image_gan import ImageGAN
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE


class ImageGANPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.images.plugin_image_gan.ImageGANPlugin
        :parts: 1

    Image (Conditional) GAN

    Args:
        n_iter: int
            Maximum number of iterations in the Generator.
        generator_nonlin: string, default 'leaky_relu'
            Nonlinearity to use in the generator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        generator_dropout: float
            Dropout value. If 0, the dropout is not used.
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

    Example:
        >>> # TODO
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_units_latent: int = 100,
        n_iter: int = 1000,
        generator_nonlin: str = "relu",
        generator_dropout: float = 0.1,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 5,
        discriminator_dropout: float = 0.1,
        # training
        lr: float = 2e-4,
        weight_decay: float = 1e-3,
        opt_betas: tuple = (0.5, 0.999),
        batch_size: int = 200,
        random_state: int = 0,
        clipping_value: int = 1,
        lambda_gradient_penalty: float = 10,
        device: Any = DEVICE,
        # early stopping
        patience: int = 5,
        patience_metric: Optional[WeightedMetrics] = None,
        n_iter_print: int = 50,
        n_iter_min: int = 100,
        plot_progress: int = False,
        # core plugin arguments
        workspace: Path = Path("workspace"),
        sampling_patience: int = 500,
        **kwargs: Any
    ) -> None:
        super().__init__(
            device=device,
            random_state=random_state,
            sampling_patience=sampling_patience,
            workspace=workspace,
            compress_dataset=False,
            **kwargs
        )
        if patience_metric is None:
            # TODO: Use something image specific
            pass

        self.n_units_latent = n_units_latent
        self.generator_nonlin = generator_nonlin
        self.n_iter = n_iter
        self.generator_dropout = generator_dropout
        self.discriminator_nonlin = discriminator_nonlin
        self.discriminator_n_iter = discriminator_n_iter
        self.discriminator_dropout = discriminator_dropout

        self.lr = lr
        self.weight_decay = weight_decay
        self.opt_betas = opt_betas

        self.batch_size = batch_size
        self.random_state = random_state
        self.clipping_value = clipping_value
        self.lambda_gradient_penalty = lambda_gradient_penalty

        self.device = device
        self.patience = patience
        self.patience_metric = patience_metric
        self.n_iter_min = n_iter_min
        self.n_iter_print = n_iter_print
        self.plot_progress = plot_progress

    @staticmethod
    def name() -> str:
        return "image_gan"

    @staticmethod
    def type() -> str:
        return "images"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            CategoricalDistribution(
                name="generator_nonlin", choices=["relu", "leaky_relu", "tanh", "elu"]
            ),
            FloatDistribution(name="generator_dropout", low=0, high=0.2),
            CategoricalDistribution(
                name="discriminator_nonlin",
                choices=["relu", "leaky_relu", "tanh", "elu"],
            ),
            FloatDistribution(name="discriminator_dropout", low=0, high=0.2),
            CategoricalDistribution(name="lr", choices=[1e-3, 2e-4, 1e-4]),
            CategoricalDistribution(name="weight_decay", choices=[1e-3, 1e-4]),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "ImageGANPlugin":
        if X.type() != "image":
            raise RuntimeError("Invalid dataloader type for image generators")

        (
            image_generator,
            image_discriminator,
        ) = suggest_image_generator_discriminator_arch(
            n_units_latent=self.n_units_latent,
            n_channels=X.info()["channels"],
            height=X.info()["height"],
            width=X.info()["width"],
            generator_dropout=self.generator_dropout,
            generator_nonlin=self.generator_nonlin,
            generator_n_residual_units=2,
            discriminator_dropout=self.discriminator_dropout,
            discriminator_nonlin=self.discriminator_nonlin,
            discriminator_n_residual_units=2,
            device=self.device,
            strategy="predefined",
        )

        self.model = ImageGAN(
            image_generator=image_generator,
            image_discriminator=image_discriminator,
            n_units_latent=self.n_units_latent,
            n_channels=X.info()["channels"],
            # generator
            generator_n_iter=self.n_iter,
            generator_lr=self.lr,
            generator_weight_decay=self.weight_decay,
            generator_opt_betas=self.opt_betas,
            generator_extra_penalties=[],
            # discriminator
            discriminator_n_iter=self.discriminator_n_iter,
            discriminator_lr=self.lr,
            discriminator_weight_decay=self.weight_decay,
            discriminator_opt_betas=self.opt_betas,
            # training
            batch_size=self.batch_size,
            random_state=self.random_state,
            clipping_value=self.clipping_value,
            lambda_gradient_penalty=self.lambda_gradient_penalty,
            device=self.device,
            n_iter_min=self.n_iter_min,
            n_iter_print=self.n_iter_print,
            plot_progress=self.plot_progress,
            patience=self.patience,
            patience_metric=self.patience_metric,
        )
        self.model.fit(X.unpack())

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> DataLoader:
        return self.model.generate(count)


plugin = ImageGANPlugin
