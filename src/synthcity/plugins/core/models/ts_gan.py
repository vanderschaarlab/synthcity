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
from synthcity.utils.reproducibility import enable_reproducible_results

# synthcity relative
from .mlp import MLP
from .ts_rnn import TimeSeriesRNN


class TimeSeriesGAN(nn.Module):
    """
    Basic TimeSeriesGAN implementation.

    Args:
        n_static_units: int,
            Number of units for the static features
        n_static_units_latent: int,
            Number of latent units for the static features
        n_temporal_units: int,
            Number of units for the temporal features
        n_temporal_window: int,
            Number of temporal sequences for each subject
        n_temporal_units_latent: int,
            Number of temporal latent units
        n_units_conditional: int = 0,
            Number of conditional units
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
        seed: int
            Seed used
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
        n_static_units: int,
        n_static_units_latent: int,
        n_temporal_units: int,
        n_temporal_window: int,
        n_temporal_units_latent: int,
        n_units_conditional: int = 0,
        generator_n_layers_hidden: int = 1,
        generator_n_units_hidden: int = 250,
        generator_nonlin: str = "leaky_relu",
        generator_static_nonlin_out: Optional[List[Tuple[str, int]]] = None,
        generator_temporal_nonlin_out: Optional[List[Tuple[str, int]]] = None,
        generator_n_iter: int = 500,
        generator_batch_norm: bool = False,
        generator_dropout: float = 0,
        generator_loss: Optional[Callable] = None,
        generator_lr: float = 2e-4,
        generator_weight_decay: float = 1e-3,
        generator_residual: bool = True,
        generator_opt_betas: tuple = (0.9, 0.999),
        generator_extra_penalties: list = [],  # "gradient_penalty", "identifiability_penalty"
        discriminator_n_layers_hidden: int = 1,
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
        seed: int = 0,
        n_iter_min: int = 100,
        clipping_value: int = 1,
        lambda_gradient_penalty: float = 10,
        lambda_identifiability_penalty: float = 0.1,
        dataloader_sampler: Optional[sampler.Sampler] = None,
        mode: str = "RNN",
        device: Any = DEVICE,
    ) -> None:
        super(TimeSeriesGAN, self).__init__()

        extra_penalty_list = ["gradient_penalty", "identifiability_penalty"]
        for penalty in discriminator_extra_penalties:
            assert (
                penalty in extra_penalty_list
            ), f"Unsupported dscriminator penalty {penalty}"
        for penalty in generator_extra_penalties:
            assert (
                penalty in extra_penalty_list
            ), f"Unsupported generator penalty {penalty}"

        log.debug(f"Training GAN on device {device}")
        self.device = device
        self.discriminator_extra_penalties = discriminator_extra_penalties
        self.generator_extra_penalties = generator_extra_penalties

        self.n_static_units = n_static_units
        self.n_static_units_latent = n_static_units_latent
        self.n_temporal_units = n_temporal_units
        self.n_temporal_units_latent = n_temporal_units_latent
        self.n_temporal_window = n_temporal_window
        self.n_units_conditional = n_units_conditional

        self.static_generator = MLP(
            task_type="regression",
            n_units_in=n_static_units_latent + n_units_conditional,
            n_units_out=n_static_units,
            n_layers_hidden=generator_n_layers_hidden,
            n_units_hidden=generator_n_units_hidden,
            nonlin=generator_nonlin,
            nonlin_out=generator_static_nonlin_out,
            n_iter=generator_n_iter,
            batch_norm=generator_batch_norm,
            dropout=generator_dropout,
            loss=generator_loss,
            seed=seed,
            lr=generator_lr,
            residual=generator_residual,
            opt_betas=generator_opt_betas,
            device=device,
        ).to(self.device)

        self.temporal_generator = TimeSeriesRNN(
            task_type="regression",
            n_static_units_in=n_static_units_latent + n_units_conditional,
            n_temporal_units_in=n_temporal_units_latent,
            output_shape=[n_temporal_window, n_temporal_units],
            window_size=n_temporal_window,
            n_static_layers_hidden=generator_n_layers_hidden,
            n_static_units_hidden=generator_n_units_hidden,
            n_temporal_layers_hidden=generator_n_layers_hidden,
            n_temporal_units_hidden=generator_n_units_hidden,
            mode=mode,
            nonlin=generator_nonlin,
            nonlin_out=generator_temporal_nonlin_out,
            n_iter=generator_n_iter,
            dropout=generator_dropout,
            loss=generator_loss,
            seed=seed,
            lr=generator_lr,
            device=device,
        ).to(self.device)

        self.discriminator = TimeSeriesRNN(
            task_type="regression",
            n_static_units_in=n_static_units + n_units_conditional,
            n_temporal_units_in=n_temporal_units,
            output_shape=[1],
            window_size=n_temporal_window,
            n_static_layers_hidden=discriminator_n_layers_hidden,
            n_static_units_hidden=discriminator_n_units_hidden,
            n_temporal_layers_hidden=discriminator_n_layers_hidden,
            n_temporal_units_hidden=discriminator_n_units_hidden,
            nonlin=discriminator_nonlin,
            mode=mode,
            nonlin_out=[("sigmoid", 1)],
            n_iter=discriminator_n_iter,
            dropout=discriminator_dropout,
            loss=discriminator_loss,
            seed=seed,
            lr=discriminator_lr,
            device=device,
        ).to(self.device)

        # training
        self.generator_n_iter = generator_n_iter
        self.discriminator_n_iter = discriminator_n_iter
        self.n_iter_print = n_iter_print
        self.n_iter_min = n_iter_min
        self.batch_size = batch_size
        self.clipping_value = clipping_value
        self.criterion = nn.BCELoss()

        self.lambda_gradient_penalty = lambda_gradient_penalty
        self.lambda_identifiability_penalty = lambda_identifiability_penalty

        self.seed = seed
        enable_reproducible_results(seed)

        self.dataloader_sampler = dataloader_sampler

    def fit(
        self,
        static_data: np.ndarray,
        temporal_data: np.ndarray,
        cond: Optional[np.ndarray] = None,
    ) -> "TimeSeriesGAN":
        static_data_t = self._check_tensor(static_data)
        temporal_data_t = self._check_tensor(temporal_data)

        condt: Optional[torch.Tensor] = None

        if self.n_units_conditional > 0:
            if cond is None:
                raise ValueError("Expecting valid conditional for training")
            if len(cond.shape) == 1:
                cond = cond.reshape(-1, 1)
            if cond.shape[1] != self.n_units_conditional:
                raise ValueError(
                    f"Expecting conditional with n_units = {self.n_units_conditional}. Got {cond.shape}"
                )
            if cond.shape[0] != static_data.shape[0]:
                raise ValueError(
                    "Expecting conditional with the same length as the dataset"
                )

            condt = self._check_tensor(cond)

        self._train(
            static_data_t,
            temporal_data_t,
            condt,
        )

        return self

    def generate(
        self, count: int, cond: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        condt: Optional[torch.Tensor] = None
        self.static_generator.eval()
        self.temporal_generator.eval()
        if cond is not None:
            condt = self._check_tensor(cond)

        static, temporal = self(count, condt)
        return static.cpu().numpy(), temporal.cpu().numpy()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(
        self,
        count: int,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cond is None and self.n_units_conditional > 0:
            # sample from the original conditional
            if self._original_cond is None:
                raise ValueError("Invalid original conditional. Provide a valid value.")
            cond_idxs = torch.randint(len(self._original_cond), (count,))
            cond = self._original_cond[cond_idxs]

        if cond is not None and len(cond.shape) == 1:
            cond = cond.reshape(-1, 1)

        if cond is not None and len(cond) != count:
            raise ValueError(
                f"cond length must match count Actual {len(cond)} Expected{count}"
            )

        static_noise = torch.randn(
            count, self.n_static_units_latent, device=self.device
        )
        static_noise = self._append_optional_cond(static_noise, cond)
        temporal_noise = torch.randn(
            count,
            self.n_temporal_window,
            self.n_temporal_units_latent,
            device=self.device,
        )

        with torch.no_grad():
            return (
                self.static_generator(static_noise).detach(),
                self.temporal_generator(static_noise, temporal_noise).detach(),
            )

    def dataloader(
        self,
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> DataLoader:
        if cond is None:
            dataset = TensorDataset(static_data, temporal_data)
        else:
            dataset = TensorDataset(static_data, temporal_data, cond)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=self.dataloader_sampler,
            pin_memory=False,
        )

    def _train_epoch_generator(
        self,
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        cond: Optional[torch.Tensor],
    ) -> float:
        # Update the G network
        self.static_generator.optimizer.zero_grad()
        self.temporal_generator.optimizer.zero_grad()

        assert len(static_data) == len(temporal_data)

        batch_size = len(static_data)

        static_noise = torch.randn(
            batch_size, self.n_static_units_latent, device=self.device
        )
        static_noise = self._append_optional_cond(static_noise, cond)
        temporal_noise = torch.randn(
            batch_size,
            self.n_temporal_window,
            self.n_temporal_units_latent,
            device=self.device,
        )

        fake_static = self.static_generator(static_noise)
        fake_static = self._append_optional_cond(fake_static, cond)
        fake_temporal = self.temporal_generator(static_noise, temporal_noise)

        output = self.discriminator(fake_static, fake_temporal).squeeze().float()

        # Calculate G's loss based on this output
        errG = self.criterion(output, torch.ones(len(fake_temporal)).to(DEVICE))

        # Calculate gradients for G
        errG.backward()

        # Update G
        if self.clipping_value > 0:
            torch.nn.utils.clip_grad_norm_(
                self.temporal_generator.parameters(), self.clipping_value
            )
            torch.nn.utils.clip_grad_norm_(
                self.static_generator.parameters(), self.clipping_value
            )
        self.temporal_generator.optimizer.step()
        self.static_generator.optimizer.step()

        # Return loss
        return errG.item()

    def _train_epoch_discriminator(
        self,
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        cond: Optional[torch.Tensor],
    ) -> float:
        # Update the D network
        errors = []

        batch_size = min(self.batch_size, len(static_data))

        for epoch in range(self.discriminator_n_iter):
            self.discriminator.zero_grad()

            # Train with all-real batch
            real_static_data = static_data.to(self.device)
            real_static_data = self._append_optional_cond(real_static_data, cond)
            real_temporal_data = temporal_data.to(self.device)

            real_labels = torch.ones(len(real_static_data)).to(self.device).squeeze()
            real_output = (
                self.discriminator(real_static_data, real_temporal_data)
                .squeeze()
                .float()
            )

            # Train with all-fake batch
            static_noise = torch.randn(
                batch_size, self.n_static_units_latent, device=self.device
            )
            static_noise = self._append_optional_cond(static_noise, cond)
            temporal_noise = torch.randn(
                batch_size,
                self.n_temporal_window,
                self.n_temporal_units_latent,
                device=self.device,
            )

            fake_static = self.static_generator(static_noise)
            fake_static = self._append_optional_cond(fake_static, cond)
            fake_temporal = self.temporal_generator(static_noise, temporal_noise)

            fake_labels = (
                torch.zeros(len(fake_static)).to(self.device).squeeze().float()
            )
            fake_output = self.discriminator(
                fake_static.detach(), fake_temporal.detach()
            ).squeeze()

            errD_real = self.criterion(real_output, real_labels)
            errD_fake = self.criterion(fake_output, fake_labels)

            errD = errD_real + errD_fake

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
    ) -> Tuple[float, float]:
        G_losses = []
        D_losses = []

        for i, data in enumerate(loader):
            cond: Optional[torch.Tensor] = None
            if self.n_units_conditional > 0:
                static_data, temporal_data, cond = data
            else:
                static_data, temporal_data = data

            D_losses.append(
                self._train_epoch_discriminator(
                    static_data,
                    temporal_data,
                    cond,
                )
            )
            G_losses.append(
                self._train_epoch_generator(
                    static_data,
                    temporal_data,
                    cond,
                )
            )

        return np.mean(G_losses), np.mean(D_losses)

    def _train(
        self,
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> "TimeSeriesGAN":
        self._original_cond = cond
        static_data = self._check_tensor(static_data).float()
        temporal_data = self._check_tensor(temporal_data).float()

        # Load Dataset
        loader = self.dataloader(static_data, temporal_data, cond)

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

    def _append_optional_cond(
        self, X: torch.Tensor, cond: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if cond is None:
            return X

        return torch.cat([X, cond], dim=1)
