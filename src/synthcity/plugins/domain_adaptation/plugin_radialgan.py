"""Conditional GAN implementation
"""
# stdlib
from typing import Any, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
import torch

# Necessary packages
from pydantic import validate_arguments
from torch import nn
from torch.utils.data import TensorDataset

# synthcity absolute
import synthcity.logger as log
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.models.mlp import MLP
from synthcity.plugins.core.models.tabular_encoder import TabularEncoder
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE
from synthcity.utils.reproducibility import clear_cache, enable_reproducible_results


class RadialGAN(nn.Module):
    """
    Basic RadialGAN implementation.

    Args:
        n_domains: int
            number of domains
        n_units_in: int
            Number of features
        generator_n_layers_hidden: int
            Number of hidden layers in the generator
        generator_n_units_hidden: int
            Number of hidden units in each layer of the Generator
        generator_nonlin: string, default 'elu'
            Nonlinearity to use in the generator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        generator_n_iter: int
            Maximum number of iterations in the Generator.
        generator_dropout: float
            Dropout value. If 0, the dropout is not used.
        discriminator_n_layers_hidden: int
            Number of hidden layers in the discriminator
        discriminator_n_units_hidden: int
            Number of hidden units in each layer of the discriminator
        discriminator_nonlin: string, default 'relu'
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
        n_iter_print: int
            Number of iterations after which to print updates and check the validation loss.
        random_state: int
            random_state used
        clipping_value: int, default 0
            Gradients clipping value. Zero disables the feature
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_domains: int,
        n_features: int,
        n_units_latent: int,
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 250,
        generator_nonlin: str = "leaky_relu",
        generator_nonlin_out: Optional[List[Tuple[str, int]]] = None,
        generator_n_iter: int = 500,
        generator_dropout: float = 0,
        generator_lr: float = 2e-4,
        generator_weight_decay: float = 1e-3,
        generator_opt_betas: tuple = (0.9, 0.999),
        discriminator_n_layers_hidden: int = 3,
        discriminator_n_units_hidden: int = 300,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_dropout: float = 0.1,
        discriminator_lr: float = 2e-4,
        discriminator_weight_decay: float = 1e-3,
        discriminator_opt_betas: tuple = (0.9, 0.999),
        batch_size: int = 64,
        n_iter_print: int = 10,
        random_state: int = 0,
        clipping_value: int = 0,
        lambda_gradient_penalty: float = 10,
        device: Any = DEVICE,
    ) -> None:
        super(RadialGAN, self).__init__()

        log.info(f"Training RadialGAN on device {device}. features = {n_features}")
        self.device = device
        self.generator_nonlin_out = generator_nonlin_out

        self.n_features = n_features
        self.n_units_latent = n_units_latent

        self.generator = MLP(
            task_type="regression",
            n_units_in=n_units_latent,
            n_units_out=n_features,
            n_layers_hidden=generator_n_layers_hidden,
            n_units_hidden=generator_n_units_hidden,
            nonlin=generator_nonlin,
            nonlin_out=generator_nonlin_out,
            n_iter=generator_n_iter,
            batch_norm=False,
            dropout=generator_dropout,
            random_state=random_state,
            lr=generator_lr,
            residual=False,
            opt_betas=generator_opt_betas,
            device=device,
        ).to(self.device)

        self.discriminator = MLP(
            task_type="regression",
            n_units_in=n_features,
            n_units_out=1,
            n_layers_hidden=discriminator_n_layers_hidden,
            n_units_hidden=discriminator_n_units_hidden,
            nonlin=discriminator_nonlin,
            nonlin_out=[("none", 1)],
            n_iter=discriminator_n_iter,
            batch_norm=False,
            dropout=discriminator_dropout,
            random_state=random_state,
            lr=discriminator_lr,
            opt_betas=discriminator_opt_betas,
            device=device,
        ).to(self.device)

        # training
        self.generator_n_iter = generator_n_iter
        self.discriminator_n_iter = discriminator_n_iter
        self.n_iter_print = n_iter_print
        self.batch_size = batch_size
        self.clipping_value = clipping_value

        self.lambda_gradient_penalty = lambda_gradient_penalty

        self.random_state = random_state
        enable_reproducible_results(random_state)

        def gen_fake_labels(X: torch.Tensor) -> torch.Tensor:
            return torch.zeros((len(X),), device=self.device)

        def gen_true_labels(X: torch.Tensor) -> torch.Tensor:
            return torch.ones((len(X),), device=self.device)

        self.fake_labels_generator = gen_fake_labels
        self.true_labels_generator = gen_true_labels

    def fit(
        self,
        X: np.ndarray,
    ) -> "RadialGAN":
        clear_cache()

        Xt = self._check_tensor(X)
        self._train(
            Xt,
        )

        return self

    def generate(self, count: int) -> np.ndarray:
        clear_cache()
        self.generator.eval()

        with torch.no_grad():
            return self(count).detach().cpu().numpy()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(
        self,
        count: int,
    ) -> torch.Tensor:
        fixed_noise = torch.randn(count, self.n_units_latent, device=self.device)
        return self.generator(fixed_noise)

    def dataloader(self, X: torch.Tensor) -> torch.utils.data.DataLoader:
        dataset = TensorDataset(X)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=self.dataloader_sampler,
            pin_memory=False,
        )

    def _train_epoch_generator(
        self,
        X: torch.Tensor,
    ) -> float:
        # Update the G network
        self.generator.optimizer.zero_grad()

        real_X = X.to(self.device)
        batch_size = len(real_X)

        noise = torch.randn(batch_size, self.n_units_latent, device=self.device)

        fake = self.generator(noise)

        output = self.discriminator(fake).squeeze().float()
        # Calculate G's loss based on this output
        errG = -torch.mean(output)

        # Calculate gradients for G
        errG.backward()

        # Update G
        if self.clipping_value > 0:
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), self.clipping_value
            )
        self.generator.optimizer.step()

        assert not torch.isnan(errG)

        # Return loss
        return errG.item()

    def _train_epoch_discriminator(
        self,
        X: torch.Tensor,
    ) -> float:
        # Update the D network
        errors = []

        batch_size = min(self.batch_size, len(X))

        for epoch in range(self.discriminator_n_iter):
            # Train with all-real batch
            real_X = X.to(self.device)

            real_labels = self.true_labels_generator(X).to(self.device).squeeze()
            real_output = self.discriminator(real_X).squeeze().float()

            # Train with all-fake batch
            noise = torch.randn(batch_size, self.n_units_latent, device=self.device)

            fake = self.generator(noise)

            fake_labels = (
                self.fake_labels_generator(fake).to(self.device).squeeze().float()
            )
            fake_output = self.discriminator(fake.detach()).squeeze()

            # Compute errors. Some fake inputs might be marked as real for privacy guarantees.

            real_real_output = real_output[(real_labels * real_output) != 0]
            real_fake_output = fake_output[(fake_labels * fake_output) != 0]
            errD_real = torch.mean(torch.concat((real_real_output, real_fake_output)))

            fake_real_output = real_output[((1 - real_labels) * real_output) != 0]
            fake_fake_output = fake_output[((1 - fake_labels) * fake_output) != 0]
            errD_fake = torch.mean(torch.concat((fake_real_output, fake_fake_output)))

            penalty = self._loss_gradient_penalty(
                real_samples=real_X,
                fake_samples=fake,
                batch_size=batch_size,
            )
            errD = -errD_real + errD_fake

            self.discriminator.optimizer.zero_grad()
            penalty.backward(retain_graph=True)
            errD.backward()

            # Update D
            if self.clipping_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), self.clipping_value
                )
            self.discriminator.optimizer.step()

            errors.append(errD.item())

        assert not np.isnan(np.mean(errors))

        return np.mean(errors)

    def _train_epoch(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> Tuple[float, float]:
        G_losses = []
        D_losses = []

        for i, data in enumerate(loader):
            X = data[0]

            D_losses.append(
                self._train_epoch_discriminator(
                    X,
                )
            )
            G_losses.append(
                self._train_epoch_generator(
                    X,
                )
            )

        return np.mean(G_losses), np.mean(D_losses)

    def _train(
        self,
        X: torch.Tensor,
    ) -> "RadialGAN":
        X = self._check_tensor(X).float()

        # Load Dataset
        loader = self.dataloader(X)

        # Train loop
        for i in range(self.generator_n_iter):
            g_loss, d_loss = self._train_epoch(
                loader,
            )
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i + 1) % self.n_iter_print == 0:
                log.debug(
                    f"[{i}/{self.generator_n_iter}]\tLoss_D: {d_loss}\tLoss_G: {g_loss}"
                )

        return self

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)

    def _loss_gradient_penalty(
        self,
        real_samples: torch.tensor,
        fake_samples: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand([batch_size, 1]).to(self.device)
        # Get random interpolation between real and fake samples
        interpolated = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        d_interpolated = self.discriminator(interpolated).squeeze()
        labels = torch.ones((len(interpolated),), device=self.device)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=labels,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()
        return self.lambda_gradient_penalty * gradient_penalty


class TabularRadialGAN(torch.nn.Module):
    """
    RadialGAN for tabular data.

    This class combines RadialGAN and tabular encoder to form a generative model for tabular data.

    Args:
        X: pd.DataFrame
            Input dataset
        domain_column: str
            The domain column
        n_units_in: int
            Number of features
        generator_n_layers_hidden: int
            Number of hidden layers in the generator
        generator_n_units_hidden: int
            Number of hidden units in each layer of the Generator
        generator_nonlin: string, default 'elu'
            Nonlinearity to use in the generator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        generator_n_iter: int
            Maximum number of iterations in the Generator.
        generator_dropout: float
            Dropout value. If 0, the dropout is not used.
        discriminator_n_layers_hidden: int
            Number of hidden layers in the discriminator
        discriminator_n_units_hidden: int
            Number of hidden units in each layer of the discriminator
        discriminator_nonlin: string, default 'relu'
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
        n_iter_print: int
            Number of iterations after which to print updates and check the validation loss.
        random_state: int
            random_state used
        clipping_value: int, default 0
            Gradients clipping value
        lambda_gradient_penalty: float
            Lambda weight for the gradient penalty
        encoder_max_clusters: int
            The max number of clusters to create for continuous columns when encoding
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        X: pd.DataFrame,
        domain_column: str,
        n_units_latent: int,
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 150,
        generator_nonlin: str = "leaky_relu",
        generator_nonlin_out_discrete: str = "softmax",
        generator_nonlin_out_continuous: str = "none",
        generator_n_iter: int = 1000,
        generator_dropout: float = 0.01,
        generator_lr: float = 1e-3,
        generator_weight_decay: float = 1e-3,
        generator_opt_betas: tuple = (0.9, 0.999),
        discriminator_n_layers_hidden: int = 3,
        discriminator_n_units_hidden: int = 300,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_dropout: float = 0.1,
        discriminator_lr: float = 1e-3,
        discriminator_weight_decay: float = 1e-3,
        discriminator_opt_betas: tuple = (0.9, 0.999),
        batch_size: int = 64,
        n_iter_print: int = 100,
        random_state: int = 0,
        n_iter_min: int = 100,
        clipping_value: int = 0,
        lambda_gradient_penalty: float = 10,
        encoder_max_clusters: int = 20,
        device: Any = DEVICE,
    ) -> None:
        super(TabularRadialGAN, self).__init__()
        self.columns = X.columns
        self.encoder = TabularEncoder(max_clusters=encoder_max_clusters).fit(X)

        self.model = RadialGAN(
            self.encoder.n_features(),
            n_units_latent=n_units_latent,
            batch_size=batch_size,
            generator_n_layers_hidden=generator_n_layers_hidden,
            generator_n_units_hidden=generator_n_units_hidden,
            generator_nonlin=generator_nonlin,
            generator_nonlin_out=self.encoder.activation_layout(
                discrete_activation=generator_nonlin_out_discrete,
                continuous_activation=generator_nonlin_out_continuous,
            ),
            generator_n_iter=generator_n_iter,
            generator_dropout=generator_dropout,
            generator_lr=generator_lr,
            generator_weight_decay=generator_weight_decay,
            generator_opt_betas=generator_opt_betas,
            discriminator_n_units_hidden=discriminator_n_units_hidden,
            discriminator_n_layers_hidden=discriminator_n_layers_hidden,
            discriminator_n_iter=discriminator_n_iter,
            discriminator_nonlin=discriminator_nonlin,
            discriminator_dropout=discriminator_dropout,
            discriminator_lr=discriminator_lr,
            discriminator_weight_decay=discriminator_weight_decay,
            discriminator_opt_betas=discriminator_opt_betas,
            lambda_gradient_penalty=lambda_gradient_penalty,
            clipping_value=clipping_value,
            n_iter_print=n_iter_print,
            random_state=random_state,
            device=device,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def encode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.transform(X)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def decode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.inverse_transform(X)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        X: pd.DataFrame,
    ) -> Any:
        X_enc = self.encode(X)

        self.model.fit(
            np.asarray(X_enc),
        )
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate(
        self,
        count: int,
    ) -> pd.DataFrame:
        samples = self(count)
        return self.decode(pd.DataFrame(samples))

    def forward(self, count: int) -> torch.Tensor:
        return self.model.generate(count)


class RadialGANPlugin(Plugin):
    """RadialGAN plugin.

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

    Example:
        >>> from synthcity.plugins import Plugins
        >>> plugin = Plugins().get("radialgan")
        >>> from sklearn.datasets import load_iris
        >>> X = load_iris()
        >>> plugin.fit(X)
        >>> plugin.generate()
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_iter: int = 2000,
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
        weight_decay: float = 1e-3,
        batch_size: int = 500,
        random_state: int = 0,
        clipping_value: int = 1,
        lambda_gradient_penalty: float = 10,
        encoder_max_clusters: int = 10,
        device: Any = DEVICE,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.generator_n_layers_hidden = generator_n_layers_hidden
        self.generator_n_units_hidden = generator_n_units_hidden
        self.generator_nonlin = generator_nonlin
        self.n_iter = n_iter
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

        self.encoder_max_clusters = encoder_max_clusters

        self.device = device

    @staticmethod
    def name() -> str:
        return "radialgan"

    @staticmethod
    def type() -> str:
        return "domain_adaptation"

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
            IntegerDistribution(name="n_iter", low=100, high=1000, step=100),
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
            CategoricalDistribution(name="batch_size", choices=[100, 200, 500]),
            IntegerDistribution(name="encoder_max_clusters", low=2, high=20),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "RadialGANPlugin":
        if X.domain() is None:
            raise ValueError("Provide the 'domain_column' info to the DataLoader")
        features = X.shape[1]

        self.model = TabularRadialGAN(
            X.dataframe(),
            n_units_latent=features,
            batch_size=self.batch_size,
            generator_n_layers_hidden=self.generator_n_layers_hidden,
            generator_n_units_hidden=self.generator_n_units_hidden,
            generator_nonlin=self.generator_nonlin,
            generator_nonlin_out_discrete="softmax",
            generator_nonlin_out_continuous="none",
            generator_lr=self.lr,
            generator_n_iter=self.n_iter,
            generator_dropout=0,
            generator_weight_decay=self.weight_decay,
            generator_opt_betas=self.generator_opt_betas,
            discriminator_n_units_hidden=self.discriminator_n_units_hidden,
            discriminator_n_layers_hidden=self.discriminator_n_layers_hidden,
            discriminator_n_iter=self.discriminator_n_iter,
            discriminator_nonlin=self.discriminator_nonlin,
            discriminator_dropout=self.discriminator_dropout,
            discriminator_lr=self.lr,
            discriminator_weight_decay=self.weight_decay,
            discriminator_opt_betas=self.discriminator_opt_betas,
            clipping_value=self.clipping_value,
            lambda_gradient_penalty=self.lambda_gradient_penalty,
            encoder_max_clusters=self.encoder_max_clusters,
            device=self.device,
        )
        self.model.fit(X.dataframe())

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> DataLoader:
        return self._safe_generate(self.model.generate, count, syn_schema)


plugin = RadialGANPlugin
