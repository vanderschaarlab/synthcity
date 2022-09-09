# stdlib
from typing import Any, Callable, List, Optional, Tuple

# third party
import numpy as np
import torch
from pydantic import validate_arguments
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, sampler

# synthcity absolute
import synthcity.logger as log
from synthcity.utils.constants import DEVICE
from synthcity.utils.reproducibility import clear_cache, enable_reproducible_results

# synthcity relative
from .mlp import MLP


class GAN(nn.Module):
    """
    Basic GAN implementation.

    Args:
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
        generator_batch_norm: bool
            Enable/disable batch norm for the generator
        generator_dropout: float
            Dropout value. If 0, the dropout is not used.
        generator_residual: bool
            Use residuals for the generator
        discriminator_n_layers_hidden: int
            Number of hidden layers in the discriminator
        discriminator_n_units_hidden: int
            Number of hidden units in each layer of the discriminator
        discriminator_nonlin: string, default 'relu'
            Nonlinearity to use in the discriminator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        discriminator_n_iter: int
            Maximum number of iterations in the discriminator.
        discriminator_batch_norm: bool
            Enable/disable batch norm for the discriminator
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
        n_iter_min: int
            Minimum number of iterations to go through before starting early stopping
        clipping_value: int, default 0
            Gradients clipping value. Zero disables the feature
        criterion: str
            Loss criterion:
                - bce: Uses BCELoss for discriminating the outputs.
                - wd: Uses the WGAN strategy for the critic.

    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_features: int,
        n_units_latent: int,
        n_units_conditional: int = 0,
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 250,
        generator_nonlin: str = "leaky_relu",
        generator_nonlin_out: Optional[List[Tuple[str, int]]] = None,
        generator_n_iter: int = 500,
        generator_batch_norm: bool = False,
        generator_dropout: float = 0,
        generator_loss: Optional[Callable] = None,
        generator_lr: float = 2e-4,
        generator_weight_decay: float = 1e-3,
        generator_residual: bool = True,
        generator_opt_betas: tuple = (0.9, 0.999),
        generator_extra_penalties: list = [],  # "gradient_penalty", "identifiability_penalty"
        discriminator_n_layers_hidden: int = 3,
        discriminator_n_units_hidden: int = 300,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_batch_norm: bool = False,
        discriminator_dropout: float = 0.1,
        discriminator_loss: Optional[Callable] = None,
        discriminator_lr: float = 2e-4,
        discriminator_weight_decay: float = 1e-3,
        discriminator_opt_betas: tuple = (0.9, 0.999),
        discriminator_extra_penalties: list = [
            "gradient_penalty"
        ],  # "gradient_penalty", "identifiability_penalty"
        batch_size: int = 64,
        n_iter_print: int = 10,
        random_state: int = 0,
        n_iter_min: int = 100,
        clipping_value: int = 0,
        lambda_gradient_penalty: float = 10,
        lambda_identifiability_penalty: float = 0.1,
        dataloader_sampler: Optional[sampler.Sampler] = None,
        device: Any = DEVICE,
    ) -> None:
        super(GAN, self).__init__()

        extra_penalty_list = ["gradient_penalty", "identifiability_penalty"]
        for penalty in discriminator_extra_penalties:
            assert (
                penalty in extra_penalty_list
            ), f"Unsupported dscriminator penalty {penalty}"
        for penalty in generator_extra_penalties:
            assert (
                penalty in extra_penalty_list
            ), f"Unsupported generator penalty {penalty}"

        log.info(f"Training GAN on device {device}. features = {n_features}")
        self.device = device
        self.discriminator_extra_penalties = discriminator_extra_penalties
        self.generator_extra_penalties = generator_extra_penalties
        self.generator_nonlin_out = generator_nonlin_out

        self.n_features = n_features
        self.n_units_latent = n_units_latent
        self.n_units_conditional = n_units_conditional

        self.generator = MLP(
            task_type="regression",
            n_units_in=n_units_latent + n_units_conditional,
            n_units_out=n_features,
            n_layers_hidden=generator_n_layers_hidden,
            n_units_hidden=generator_n_units_hidden,
            nonlin=generator_nonlin,
            nonlin_out=generator_nonlin_out,
            n_iter=generator_n_iter,
            batch_norm=generator_batch_norm,
            dropout=generator_dropout,
            loss=generator_loss,
            random_state=random_state,
            lr=generator_lr,
            residual=generator_residual,
            opt_betas=generator_opt_betas,
            device=device,
        ).to(self.device)

        self.discriminator = MLP(
            task_type="regression",
            n_units_in=n_features + n_units_conditional,
            n_units_out=1,
            n_layers_hidden=discriminator_n_layers_hidden,
            n_units_hidden=discriminator_n_units_hidden,
            nonlin=discriminator_nonlin,
            nonlin_out=[("none", 1)],
            n_iter=discriminator_n_iter,
            batch_norm=discriminator_batch_norm,
            dropout=discriminator_dropout,
            loss=discriminator_loss,
            random_state=random_state,
            lr=discriminator_lr,
            opt_betas=discriminator_opt_betas,
            device=device,
        ).to(self.device)

        # training
        self.generator_n_iter = generator_n_iter
        self.discriminator_n_iter = discriminator_n_iter
        self.n_iter_print = n_iter_print
        self.n_iter_min = n_iter_min
        self.batch_size = batch_size
        self.clipping_value = clipping_value

        self.lambda_gradient_penalty = lambda_gradient_penalty
        self.lambda_identifiability_penalty = lambda_identifiability_penalty

        self.random_state = random_state
        enable_reproducible_results(random_state)

        def gen_fake_labels(X: torch.Tensor) -> torch.Tensor:
            return torch.zeros((len(X),), device=self.device)

        def gen_true_labels(X: torch.Tensor) -> torch.Tensor:
            return torch.ones((len(X),), device=self.device)

        self.fake_labels_generator = gen_fake_labels
        self.true_labels_generator = gen_true_labels
        self.dataloader_sampler = dataloader_sampler

    def fit(
        self,
        X: np.ndarray,
        cond: Optional[np.ndarray] = None,
        fake_labels_generator: Optional[Callable] = None,
        true_labels_generator: Optional[Callable] = None,
    ) -> "GAN":
        clear_cache()

        Xt = self._check_tensor(X)
        condt: Optional[torch.Tensor] = None

        if self.n_units_conditional > 0:
            if cond is None:
                raise ValueError("Expecting valid conditional for training")
            if len(cond.shape) == 1:
                cond = cond.reshape(-1, 1)
            if cond.shape[1] != self.n_units_conditional:
                raise ValueError(
                    "Expecting conditional with n_units = {self.n_units_conditional}"
                )
            if cond.shape[0] != X.shape[0]:
                raise ValueError(
                    "Expecting conditional with the same length as the dataset"
                )

            condt = self._check_tensor(cond)

        self._train(
            Xt,
            condt,
            fake_labels_generator=fake_labels_generator,
            true_labels_generator=true_labels_generator,
        )

        return self

    def generate(self, count: int, cond: Optional[np.ndarray] = None) -> np.ndarray:
        clear_cache()
        self.generator.eval()

        condt: Optional[torch.Tensor] = None
        if cond is not None:
            condt = self._check_tensor(cond)
        with torch.no_grad():
            return self(count, condt).detach().cpu().numpy()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(
        self,
        count: int,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cond is None and self.n_units_conditional > 0:
            # sample from the original conditional
            if self._original_cond is None:
                raise ValueError("Invalid original conditional. Provide a valid value.")
            cond_idxs = torch.randint(len(self._original_cond), (count,))
            cond = self._original_cond[cond_idxs]

        if cond is not None and len(cond.shape) == 1:
            cond = cond.reshape(-1, 1)

        if cond is not None and len(cond) != count:
            raise ValueError("cond length must match count")

        fixed_noise = torch.randn(count, self.n_units_latent, device=self.device)
        fixed_noise = self._append_optional_cond(fixed_noise, cond)

        return self.generator(fixed_noise)

    def dataloader(
        self, X: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> DataLoader:
        if cond is None:
            dataset = TensorDataset(X)
        else:
            dataset = TensorDataset(X, cond)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=self.dataloader_sampler,
            pin_memory=False,
        )

    def _train_epoch_generator(
        self,
        X: torch.Tensor,
        cond: Optional[torch.Tensor],
        fake_labels_generator: Callable,
        true_labels_generator: Callable,
    ) -> float:
        # Update the G network
        self.generator.optimizer.zero_grad()

        real_X_raw = X.to(self.device)
        real_X = self._append_optional_cond(real_X_raw, cond)
        batch_size = len(real_X)

        noise = torch.randn(batch_size, self.n_units_latent, device=self.device)
        noise = self._append_optional_cond(noise, cond)

        fake_raw = self.generator(noise)
        fake = self._append_optional_cond(fake_raw, cond)

        output = self.discriminator(fake).squeeze().float()
        # Calculate G's loss based on this output
        errG = -torch.mean(output)
        if self.generator_nonlin_out is not None:
            errG += self._loss_cond(real_X_raw, fake_raw, self.generator_nonlin_out)

        errG -= self._extra_penalties(
            self.generator_extra_penalties,
            real_samples=real_X,
            fake_samples=fake,
            batch_size=batch_size,
            fake_labels_generator=fake_labels_generator,
            true_labels_generator=true_labels_generator,
        )

        # Calculate gradients for G
        errG.backward()

        # Update G
        if self.clipping_value > 0:
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), self.clipping_value
            )
        self.generator.optimizer.step()

        # Return loss
        return errG.item()

    def _train_epoch_discriminator(
        self,
        X: torch.Tensor,
        cond: Optional[torch.Tensor],
        fake_labels_generator: Callable,
        true_labels_generator: Callable,
    ) -> float:
        # Update the D network
        errors = []

        batch_size = min(self.batch_size, len(X))

        for epoch in range(self.discriminator_n_iter):
            self.discriminator.zero_grad()

            # Train with all-real batch
            real_X = X.to(self.device)
            real_X = self._append_optional_cond(real_X, cond)

            real_labels = true_labels_generator(X).to(self.device).squeeze()
            real_output = self.discriminator(real_X).squeeze().float()

            # Train with all-fake batch
            noise = torch.randn(batch_size, self.n_units_latent, device=self.device)
            noise = self._append_optional_cond(noise, cond)

            fake = self.generator(noise)
            fake = self._append_optional_cond(fake, cond)

            fake_labels = fake_labels_generator(fake).to(self.device).squeeze().float()
            fake_output = self.discriminator(fake.detach()).squeeze()

            # Compute errors. Some fake inputs might be marked as real for privacy guarantees.

            real_real_output = real_output[(real_labels * real_output) != 0]
            real_fake_output = fake_output[(fake_labels * fake_output) != 0]
            errD_real = torch.mean(torch.concat((real_real_output, real_fake_output)))

            fake_real_output = real_output[((1 - real_labels) * real_output) != 0]
            fake_fake_output = fake_output[((1 - fake_labels) * fake_output) != 0]
            errD_fake = torch.mean(torch.concat((fake_real_output, fake_fake_output)))

            errD = -errD_real + errD_fake
            errD += self._extra_penalties(
                self.discriminator_extra_penalties,
                real_samples=real_X,
                fake_samples=fake,
                batch_size=batch_size,
                fake_labels_generator=fake_labels_generator,
                true_labels_generator=true_labels_generator,
            )

            errD.backward()
            # Update D
            if self.clipping_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), self.clipping_value
                )
            self.discriminator.optimizer.step()

            errors.append(errD.item())

        return np.mean(errors)

    def _train_epoch(
        self,
        loader: DataLoader,
        fake_labels_generator: Optional[Callable] = None,
        true_labels_generator: Optional[Callable] = None,
    ) -> Tuple[float, float]:
        if fake_labels_generator is None:
            fake_labels_generator = self.fake_labels_generator
        if true_labels_generator is None:
            true_labels_generator = self.true_labels_generator

        G_losses = []
        D_losses = []

        for i, data in enumerate(loader):
            cond: Optional[torch.Tensor] = None
            if self.n_units_conditional > 0:
                X, cond = data
            else:
                X = data[0]

            D_losses.append(
                self._train_epoch_discriminator(
                    X,
                    cond,
                    fake_labels_generator=fake_labels_generator,
                    true_labels_generator=true_labels_generator,
                )
            )
            G_losses.append(
                self._train_epoch_generator(
                    X,
                    cond,
                    fake_labels_generator=fake_labels_generator,
                    true_labels_generator=true_labels_generator,
                )
            )

        return np.mean(G_losses), np.mean(D_losses)

    def _train(
        self,
        X: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        fake_labels_generator: Optional[Callable] = None,
        true_labels_generator: Optional[Callable] = None,
    ) -> "GAN":
        self._original_cond = cond
        X = self._check_tensor(X).float()

        # Load Dataset
        loader = self.dataloader(X, cond)

        # Train loop
        for i in range(self.generator_n_iter):
            g_loss, d_loss = self._train_epoch(
                loader,
                fake_labels_generator=fake_labels_generator,
                true_labels_generator=true_labels_generator,
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

    def _extra_penalties(
        self,
        penalties: list,
        real_samples: torch.tensor,
        fake_samples: torch.Tensor,
        batch_size: int,
        fake_labels_generator: Callable,
        true_labels_generator: Callable,
    ) -> torch.Tensor:
        """Calculates additional penalties for the training"""
        err: torch.Tensor = 0
        for penalty in penalties:
            if penalty == "gradient_penalty":
                err += self._loss_gradient_penalty(
                    real_samples=real_samples,
                    fake_samples=fake_samples,
                    batch_size=batch_size,
                    fake_labels_generator=fake_labels_generator,
                    true_labels_generator=true_labels_generator,
                )
            elif penalty == "identifiability_penalty":
                err += self._loss_identifiability_penalty(
                    real_samples=real_samples,
                    fake_samples=fake_samples,
                )
        return err

    def _loss_gradient_penalty(
        self,
        real_samples: torch.tensor,
        fake_samples: torch.Tensor,
        batch_size: int,
        fake_labels_generator: Callable,
        true_labels_generator: Callable,
    ) -> torch.Tensor:
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand([batch_size, 1]).to(self.device)
        # Get random interpolation between real and fake samples
        interpolated = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        d_interpolated = self.discriminator(interpolated).squeeze()
        labels = true_labels_generator(interpolated).to(self.device).squeeze()
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=labels,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return self.lambda_gradient_penalty * gradient_penalty

    def _loss_identifiability_penalty(
        self,
        real_samples: torch.tensor,
        fake_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the identifiability penalty"""
        return self.lambda_identifiability_penalty * torch.sqrt(
            nn.MSELoss()(real_samples, fake_samples)
        )

    def _loss_cond(
        self,
        real_samples: torch.tensor,
        fake_samples: torch.Tensor,
        activations: List,
    ) -> torch.Tensor:
        st = 0
        loss = 0
        for activation, length in activations:
            if activation == "softmax":
                loss += nn.functional.cross_entropy(
                    fake_samples[:, st : st + length],
                    torch.argmax(real_samples[:, st : st + length], dim=1),
                )

            st += length
        return loss

    def _append_optional_cond(
        self, X: torch.Tensor, cond: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if cond is None:
            return X

        return torch.cat([X, cond], dim=1)
