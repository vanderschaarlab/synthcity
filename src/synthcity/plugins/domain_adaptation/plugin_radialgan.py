"""
Reference: Yoon, Jinsung and Jordon, James and van der Schaar, Mihaela
    "RadialGAN: Leveraging multiple datasets to improve target-specific predictive models using Generative Adversarial Networks"

"""
# stdlib
from typing import Any, Dict, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
import torch

# Necessary packages
from pydantic import validate_arguments
from torch import nn
from torch.utils.data import TensorDataset
from tqdm import tqdm

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
    .. inheritance-diagram:: synthcity.plugins.domain_adaptation.plugin_radialgan.RadialGAN
        :parts: 1


    RadialGAN implementation: Leveraging multiple datasets to improve target-specific predictive models using Generative Adversarial Networks.

    Args:
        domains: List[int]
            List of domains
        n_features: int
            Number of features
        n_units_latent: int
            Number of hidden units
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
        domains: List[int],
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
        dataloader_sampler: Any = None,
    ) -> None:
        super(RadialGAN, self).__init__()

        self.domains = list(np.unique(domains))
        if len(self.domains) == 0:
            raise ValueError("Expected a positive number of domains")

        log.info(f"Training RadialGAN on device {device}. features = {n_features}")
        self.device = device
        self.generator_nonlin_out = generator_nonlin_out
        self.dataloader_sampler = dataloader_sampler

        self.n_features = n_features
        self.n_units_latent = n_units_latent

        self.generators = {}  # Domain generators
        self.discriminators = {}  # Domain discriminators
        self.mappers = {}  # Neural nets converting data to the current domain

        for domain in domains:
            self.generators[domain] = MLP(
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

        for domain in domains:
            self.discriminators[domain] = MLP(
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

        for domain in domains:
            self.mappers[domain] = MLP(
                task_type="regression",
                n_units_in=n_features,
                n_units_out=n_features,
                n_layers_hidden=generator_n_layers_hidden,
                n_units_hidden=generator_n_units_hidden,
                nonlin=generator_nonlin,
                nonlin_out=generator_nonlin_out,
                batch_norm=False,
                dropout=0,
                random_state=random_state,
                lr=generator_lr,
                opt_betas=generator_opt_betas,
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

        self.domain_weights: Optional[Dict[int, float]] = None

    def fit(
        self,
        X: np.ndarray,
        domains: np.ndarray,
    ) -> "RadialGAN":
        clear_cache()

        domain_keys, domain_counts = np.unique(domains, return_counts=True)
        domain_counts = domain_counts.astype(float)
        domain_counts /= np.sum(domain_counts) + 1e-8
        self.domain_weights = {k: v for k, v in zip(domain_keys, domain_counts)}

        Xt = self._check_tensor(X)
        domainst = self._check_tensor(domains)
        self._train(
            Xt,
            domainst,
        )

        return self

    def generate(
        self, count: int, domains: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        clear_cache()
        for domain in self.generators:
            self.generators[domain].eval()
            self.mappers[domain].eval()

        with torch.no_grad():
            samples, domains = self(count, domains)
            samples = samples.detach().cpu().numpy()
            domains = np.asarray(domains)

            return samples, domains

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(
        self, count: int, domains: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, List]:
        if domains is None:
            domains = self.domains

        if self.domain_weights is None:
            raise ValueError("Invalid domain weights")

        batch_per_domain = count // len(domains) + 1
        out = torch.tensor([]).to(self.device)
        out_domains = []
        for target_domain in domains:
            for src_domain in self.domain_weights:
                src_batch_size = int(batch_per_domain * self.domain_weights[src_domain])
                fixed_noise = torch.randn(
                    src_batch_size, self.n_units_latent, device=self.device
                )
                domain_generated = self.generators[src_domain](fixed_noise)
                if src_domain != target_domain:
                    domain_generated = self.mappers[target_domain](domain_generated)

                out = torch.concat([out, domain_generated])
                out_domains.extend([target_domain] * len(domain_generated))

        return out, out_domains

    def dataloader(
        self, X: torch.Tensor, domains: torch.Tensor
    ) -> torch.utils.data.DataLoader:
        dataset = TensorDataset(X, domains)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=self.dataloader_sampler,
            pin_memory=False,
        )

    def _train_epoch_mapper(
        self,
        domain: int,
        X: torch.Tensor,
    ) -> float:
        batch_size = len(X)
        if batch_size == 0:
            return 0
        # Update the M network
        self.mappers[domain].optimizer.zero_grad()

        real_X = X.to(self.device)

        noise = torch.randn(batch_size, self.n_units_latent, device=self.device)

        errs = []
        for other_domain in self.domains:
            if other_domain == domain:
                continue

            fake = self.generators[other_domain](
                noise
            )  # generate fake data for <other_domain>
            fake = self.mappers[domain](fake)  # remap data to domain <domain>

            # Calculate M's loss based on this output
            errM = nn.MSELoss()(fake, real_X)

            errs.append(errM)

        # Calculate gradients for M
        errM = 0.1 * torch.sqrt(torch.stack(errs)).mean()
        errM.backward()

        # Update M
        if self.clipping_value > 0:
            torch.nn.utils.clip_grad_norm_(
                self.mappers[domain].parameters(), self.clipping_value
            )
        self.mappers[domain].optimizer.step()

        if torch.isnan(errM):
            raise RuntimeError("NaNs detected in the mapper loss")

        # Return loss
        return errM.item()

    def _train_epoch_generator(
        self,
        domain: int,
        X: torch.Tensor,
    ) -> float:
        batch_size = len(X)
        if batch_size == 0:
            return 0
        # Update the G network
        self.generators[domain].optimizer.zero_grad()

        noise = torch.randn(batch_size, self.n_units_latent, device=self.device)

        errs = []
        for other_domain in self.domains:
            fake = self.generators[domain](noise)  # generate fake data
            if other_domain != domain:
                fake = self.mappers[other_domain](
                    fake
                )  # remap data to domain <other_domain>

            output = self.discriminators[other_domain](fake).squeeze().float()
            # Calculate G's loss based on this output
            errG = -torch.mean(output)

            errs.append(errG)

        # Calculate gradients for G
        errG = torch.stack(errs).mean()
        errG.backward()

        # Update G
        if self.clipping_value > 0:
            torch.nn.utils.clip_grad_norm_(
                self.generators[domain].parameters(), self.clipping_value
            )
        self.generators[domain].optimizer.step()

        if torch.isnan(errG):
            raise RuntimeError("NaNs detected in the generator loss")

        # Return loss
        return errG.item()

    def _train_epoch_discriminator(
        self,
        domain: int,
        X: torch.Tensor,
    ) -> float:
        # Update the D network
        errors = []

        batch_size = min(self.batch_size, len(X))
        if batch_size == 0:
            return 0

        for epoch in range(self.discriminator_n_iter):
            # Train with all-real batch
            real_X = X.to(self.device)

            real_labels = self.true_labels_generator(X).to(self.device).squeeze()
            real_output = self.discriminators[domain](real_X).squeeze().float()

            # Train with all-fake batch
            errD_fakes = []
            penalties = []
            for other_domain in self.domains:
                noise = torch.randn(batch_size, self.n_units_latent, device=self.device)

                fake = self.generators[other_domain](
                    noise
                )  # generate fake data for domain <other_domain>

                if other_domain != domain:
                    fake = self.mappers[domain](
                        fake
                    )  # remap the generate data to domain <domain>

                fake_labels = (
                    self.fake_labels_generator(fake).to(self.device).squeeze().float()
                )
                fake_output = self.discriminators[domain](fake.detach()).squeeze()

                # Compute errors. Some fake inputs might be marked as real for privacy guarantees.

                real_real_output = real_output[(real_labels * real_output) != 0]
                real_fake_output = fake_output[(fake_labels * fake_output) != 0]
                errD_real = torch.mean(
                    torch.concat((real_real_output, real_fake_output))
                )

                fake_real_output = real_output[((1 - real_labels) * real_output) != 0]
                fake_fake_output = fake_output[((1 - fake_labels) * fake_output) != 0]

                errD_fake = torch.mean(
                    torch.concat((fake_real_output, fake_fake_output))
                )

                penalty = self._loss_gradient_penalty(
                    domain=domain,
                    real_samples=real_X,
                    fake_samples=fake,
                    batch_size=batch_size,
                )
                errD_fakes.append(errD_fake)
                penalties.append(penalty)

            errD_fake = torch.stack(errD_fakes)
            penalty = torch.stack(penalties)
            errD = -errD_real + errD_fake.mean()

            self.discriminators[domain].optimizer.zero_grad()
            torch.mean(penalty).backward(retain_graph=True)
            errD.backward()

            # Update D
            if self.clipping_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.discriminators[domain].parameters(), self.clipping_value
                )
            self.discriminators[domain].optimizer.step()

            errors.append(errD.item())

        if np.isnan(np.mean(errors)):
            raise RuntimeError("NaNs detected in the discriminator loss")

        return np.mean(errors)

    def _train_epoch(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> Tuple[float, float, float]:
        G_losses = []
        D_losses = []
        M_losses = []

        for i, data in enumerate(loader):
            X, X_domains = data

            for domain in self.domains:
                X_batch = X[X_domains == domain]
                D_losses.append(
                    self._train_epoch_discriminator(
                        domain,
                        X_batch,
                    )
                )
                G_losses.append(
                    self._train_epoch_generator(
                        domain,
                        X_batch,
                    )
                )
                M_losses.append(
                    self._train_epoch_mapper(
                        domain,
                        X_batch,
                    )
                )

        return np.mean(G_losses), np.mean(D_losses), np.mean(M_losses)

    def _train(
        self,
        X: torch.Tensor,
        domains: torch.Tensor,
    ) -> "RadialGAN":
        X = self._check_tensor(X).float()
        domains = self._check_tensor(domains).long()

        # Load Dataset
        loader = self.dataloader(X, domains)

        # Train loop
        for i in tqdm(range(self.generator_n_iter)):
            g_loss, d_loss, m_loss = self._train_epoch(
                loader,
            )
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i + 1) % self.n_iter_print == 0:
                log.debug(
                    f"[{i}/{self.generator_n_iter}]\tLoss_D: {d_loss}\tLoss_G: {g_loss}\tLoss_M: {m_loss}"
                )

        return self

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)

    def _loss_gradient_penalty(
        self,
        domain: int,
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
        if batch_size > 1:
            d_interpolated = self.discriminators[domain](interpolated).squeeze()
        else:
            d_interpolated = self.discriminators[domain](interpolated)[0, :]

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
        domains = list(X[domain_column].unique())
        X = X.drop(columns=[domain_column])

        self.columns = X.columns
        self.domain_column = domain_column

        self.encoder = TabularEncoder(max_clusters=encoder_max_clusters).fit(X)

        self.model = RadialGAN(
            domains=domains,
            n_features=self.encoder.n_features(),
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
        domains = X[self.domain_column]
        X_enc = self.encode(X)

        self.model.fit(
            np.asarray(X_enc),
            np.asarray(domains),
        )
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate(
        self,
        count: int,
        domains: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        samples, domains = self(count, domains=domains)
        samples = self.decode(pd.DataFrame(samples))
        samples[self.domain_column] = domains

        return samples

    def forward(self, count: int, domains: Optional[List[int]] = None) -> torch.Tensor:
        return self.model.generate(count, domains=domains)


class RadialGANPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.domain_adaptation.plugin_radialgan.RadialGANPlugin
        :parts: 1


    RadialGAN PyTorch implementation: Leveraging multiple datasets to improve target-specific predictive models using Generative Adversarial Networks.


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
        >>> import numpy as np
        >>> from sklearn.datasets import load_iris
        >>> from synthcity.plugins import Plugins
        >>> from synthcity.plugins.core.dataloader import GenericDataLoader
        >>>
        >>> X, y = load_iris(as_frame = True, return_X_y = True)
        >>> X["target"] = y
        >>> X["domain"] = np.random.choice([0, 1], len(X)) # simulate domains
        >>> dataloader = GenericDataLoader(X, domain_column="domain")
        >>>
        >>> plugin = Plugins().get("radialgan", n_iter = 100)
        >>> plugin.fit(dataloader)
        >>>
        >>> plugin.generate(50)

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

        self.model = TabularRadialGAN(
            X.dataframe(),
            domain_column=X.domain(),
            n_units_latent=self.generator_n_units_hidden,
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

    def _generate(
        self,
        count: int,
        syn_schema: Schema,
        domains: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> DataLoader:
        def _model_generate(count: int) -> pd.DataFrame:
            return self.model.generate(count, domains)

        return self._safe_generate(_model_generate, count, syn_schema)


plugin = RadialGANPlugin
