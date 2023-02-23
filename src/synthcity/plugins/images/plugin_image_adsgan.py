# stdlib
from pathlib import Path
from typing import Any, List, Optional

# third party
import numpy as np
import torch

# Necessary packages
from pydantic import validate_arguments
from torch import nn

# synthcity absolute
import synthcity.logger as log
from synthcity.metrics.weighted_metrics import WeightedMetrics
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.dataset import TensorDataset
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
)
from synthcity.plugins.core.models.convnet import (
    suggest_image_classifier_arch,
    suggest_image_generator_discriminator_arch,
)
from synthcity.plugins.core.models.image_gan import ImageGAN
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE


class ImageAdsGANPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.images.plugin_image_adsgan.ImageAdsGANPlugin
        :parts: 1

    Image AdsGAN - Anonymization through Data Synthesis using Generative Adversarial Networks.

    Args:
        n_units_latent: int
            The noise units size used by the generator.
        n_iter: int
            Maximum number of iterations in the Generator.
        generator_nonlin: string, default 'leaky_relu'
            Nonlinearity to use in the generator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        generator_dropout: float
            Dropout value. If 0, the dropout is not used.
        generator_n_residual_units: int
            The number of convolutions in residual units for the generator, 0 means no residual units
        discriminator_nonlin: string, default 'leaky_relu'
            Nonlinearity to use in the discriminator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        discriminator_n_iter: int
            Maximum number of iterations in the discriminator.
        discriminator_dropout: float
            Dropout value for the discriminator. If 0, the dropout is not used.
        discriminator_n_residual_units: int
            The number of convolutions in residual units for the discriminator, 0 means no residual units
        # training parameters
        lr: float
            learning rate for optimizer
        weight_decay: float
            l2 (ridge) penalty for the weights.
        batch_size: int
            Batch size
        random_state: int
            random seed to use
        clipping_value: int, default 0
            Gradients clipping value. Zero disables the feature
        lambda_gradient_penalty: float = 10
            Weight for the gradient penalty
        lambda_identifiability_penalty: float = 0.1
            Weight for the identifiability penalty
        device: torch device
            Device: cpu or cuda
        plot_progress: bool
            Plot some synthetic samples every `n_iter_print`
        # early stopping
        n_iter_print: int
            Number of iterations after which to print updates and check the validation loss.
        n_iter_min: int
            Minimum number of iterations to go through before starting early stopping
        early_stopping: bool
            Evaluate the quality of the synthetic data using `patience_metric`, and stop after `patience` iteration with no improvement.
        patience: int
            Max number of iterations without any improvement before training early stopping is trigged.
        patience_metric: Optional[WeightedMetrics]
            If not None, the metric is used for evaluation the criterion for training early stopping.
        # Core Plugin arguments
        workspace: Path.
            Optional Path for caching intermediary results.

    Example:
        >>> from torchvision import datasets
        >>> from synthcity.plugins import Plugins
        >>> from synthcity.plugins.core.dataloader import ImageDataLoader
        >>>
        >>> model = Plugins().get("image_adsgan", n_iter = 10)
        >>>
        >>> dataset = datasets.MNIST(".", download=True)
        >>> dataloader = ImageDataLoader(dataset).sample(100)
        >>>
        >>> model.fit(dataloader)
        >>>
        >>> X_gen = model.generate(50)
        >>> assert len(X_gen) == 50
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_units_latent: int = 100,
        n_iter: int = 1000,
        generator_nonlin: str = "relu",
        generator_dropout: float = 0.1,
        generator_n_residual_units: int = 2,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 5,
        discriminator_dropout: float = 0.1,
        discriminator_n_residual_units: int = 2,
        # training
        lr: float = 2e-4,
        weight_decay: float = 1e-3,
        opt_betas: tuple = (0.5, 0.999),
        batch_size: int = 200,
        random_state: int = 0,
        clipping_value: int = 1,
        lambda_gradient_penalty: float = 10,
        lambda_identifiability_penalty: float = 0.1,
        device: Any = DEVICE,
        # early stopping
        patience: int = 5,
        patience_metric: Optional[WeightedMetrics] = None,
        n_iter_print: int = 50,
        n_iter_min: int = 100,
        plot_progress: int = False,
        early_stopping: bool = True,
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
            patience_metric = WeightedMetrics(
                metrics=[("detection", "detection_mlp")],
                weights=[1],
                workspace=workspace,
            )

        self.n_units_latent = n_units_latent
        self.n_iter = n_iter
        self.generator_nonlin = generator_nonlin
        self.generator_dropout = generator_dropout
        self.generator_n_residual_units = generator_n_residual_units
        self.discriminator_nonlin = discriminator_nonlin
        self.discriminator_n_iter = discriminator_n_iter
        self.discriminator_dropout = discriminator_dropout
        self.discriminator_n_residual_units = discriminator_n_residual_units

        self.lr = lr
        self.weight_decay = weight_decay
        self.opt_betas = opt_betas

        self.batch_size = batch_size
        self.random_state = random_state
        self.clipping_value = clipping_value
        self.lambda_gradient_penalty = lambda_gradient_penalty
        self.lambda_identifiability_penalty = lambda_identifiability_penalty

        self.device = device
        self.patience = patience
        self.patience_metric = patience_metric
        self.early_stopping = early_stopping
        self.n_iter_min = n_iter_min
        self.n_iter_print = n_iter_print
        self.plot_progress = plot_progress

    @staticmethod
    def name() -> str:
        return "image_adsgan"

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

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "ImageAdsGANPlugin":
        if X.type() != "images":
            raise RuntimeError("Invalid dataloader type for image generators")

        labels = X.unpack().labels()
        self.classes = np.unique(labels)

        cond = labels
        if "cond" in kwargs:
            cond = kwargs["cond"]

        cond = self._prepare_cond(cond)

        # synthetic images
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
            generator_n_residual_units=self.generator_n_residual_units,
            discriminator_dropout=self.discriminator_dropout,
            discriminator_nonlin=self.discriminator_nonlin,
            discriminator_n_residual_units=self.discriminator_n_residual_units,
            device=self.device,
            strategy="predefined",
            cond=cond,
            cond_embedding_n_units_hidden=self.n_units_latent,
        )

        log.debug("Training the image generator")
        self.image_generator = ImageGAN(
            image_generator=image_generator,
            image_discriminator=image_discriminator,
            n_units_latent=self.n_units_latent,
            n_channels=X.info()["channels"],
            # generator
            generator_n_iter=self.n_iter,
            generator_lr=self.lr,
            generator_weight_decay=self.weight_decay,
            generator_opt_betas=self.opt_betas,
            generator_extra_penalties=["identifiability_penalty"],
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
            lambda_identifiability_penalty=self.lambda_identifiability_penalty,
            device=self.device,
            n_iter_min=self.n_iter_min,
            n_iter_print=self.n_iter_print,
            plot_progress=self.plot_progress,
            patience=self.patience,
            patience_metric=self.patience_metric,
        )
        self.image_generator.fit(X.unpack(), cond=cond)

        # synthetic labels
        self.label_generator: Optional[nn.Module] = None

        if labels is not None:  # TODO: handle regression
            log.debug("Training the labels generator")
            self.label_generator = suggest_image_classifier_arch(
                n_channels=X.info()["channels"],
                height=X.info()["height"],
                width=X.info()["width"],
                classes=len(np.unique(labels)),
                n_residual_units=self.generator_n_residual_units,
                nonlin=self.generator_nonlin,
                dropout=self.generator_dropout,
                last_nonlin="softmax",
                device=self.device,
                strategy="predefined",
                # training
                lr=self.lr,
                weight_decay=self.weight_decay,
                opt_betas=self.opt_betas,
                n_iter=self.n_iter,
                batch_size=self.batch_size,
                n_iter_print=self.n_iter_print,
                random_state=self.random_state,
                patience=self.patience,
                n_iter_min=self.n_iter_min,
                clipping_value=self.clipping_value,
                early_stopping=self.early_stopping,
            )
            self.label_generator.fit(X.unpack())

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> DataLoader:
        def _sample(count: int) -> TensorDataset:
            cond: Optional[torch.Tensor] = None
            if "cond" in kwargs:
                cond = self._prepare_cond(kwargs["cond"])
            elif self.classes is not None:
                cond = np.random.choice(self.classes, count)
                cond = torch.from_numpy(cond).to(self.device)

            sampled_images = self.image_generator.generate(count, cond=cond)
            sampled_labels: Optional[torch.Tensor] = None
            if self.label_generator is not None:
                sampled_labels = self.label_generator.predict(sampled_images)

            return TensorDataset(images=sampled_images, targets=sampled_labels)

        return self._safe_generate_images(_sample, count, syn_schema)

    def _prepare_cond(self, cond: Any) -> Optional[torch.Tensor]:
        if cond is None:
            return None

        cond = np.asarray(cond)
        if len(cond.shape) == 1:
            cond = cond.reshape(-1, 1)

        return torch.from_numpy(cond).to(self.device)


plugin = ImageAdsGANPlugin
