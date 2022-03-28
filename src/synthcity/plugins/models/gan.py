# stdlib
from typing import Callable, List, Optional, Tuple

# third party
import numpy as np
import torch
from pydantic import validate_arguments
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# synthcity absolute
import synthcity.logger as log

# synthcity relative
from .mlp import MLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        seed: int
            Seed used
        n_iter_min: int
            Minimum number of iterations to go through before starting early stopping
        clipping_value: int, default 1
            Gradients clipping value
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_features: int,
        n_units_latent: int,
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
        discriminator_n_layers_hidden: int = 3,
        discriminator_n_units_hidden: int = 300,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_batch_norm: bool = False,
        discriminator_dropout: float = 0.1,
        discriminator_loss: Optional[Callable] = None,
        discriminator_lr: float = 2e-4,
        discriminator_weight_decay: float = 1e-3,
        batch_size: int = 64,
        n_iter_print: int = 10,
        seed: int = 0,
        n_iter_min: int = 100,
        clipping_value: int = 1,
    ) -> None:
        super(GAN, self).__init__()

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
            batch_norm=generator_batch_norm,
            dropout=generator_dropout,
            loss=generator_loss,
            seed=seed,
            lr=generator_lr,
            residual=generator_residual,
        ).to(DEVICE)

        self.discriminator = MLP(
            task_type="classification",
            n_units_in=n_features,
            n_units_out=1,
            n_layers_hidden=discriminator_n_layers_hidden,
            n_units_hidden=discriminator_n_units_hidden,
            nonlin=discriminator_nonlin,
            nonlin_out=[("sigmoid", 1)],
            n_iter=discriminator_n_iter,
            batch_norm=discriminator_batch_norm,
            dropout=discriminator_dropout,
            loss=discriminator_loss,
            seed=seed,
            lr=discriminator_lr,
        ).to(DEVICE)

        # training
        self.generator_n_iter = generator_n_iter
        self.discriminator_n_iter = discriminator_n_iter
        self.seed = seed
        self.n_iter_print = n_iter_print
        self.n_iter_min = n_iter_min
        self.batch_size = batch_size
        self.clipping_value = clipping_value
        self.criterion = nn.BCELoss()

        torch.manual_seed(seed)

        def gen_fake_labels(X: torch.Tensor) -> torch.Tensor:
            return torch.zeros((len(X),), device=DEVICE)

        def gen_true_labels(X: torch.Tensor) -> torch.Tensor:
            return torch.ones((len(X),), device=DEVICE)

        self.fake_labels_generator = gen_fake_labels
        self.true_labels_generator = gen_true_labels

    def fit(
        self,
        X: np.ndarray,
        fake_labels_generator: Optional[Callable] = None,
        true_labels_generator: Optional[Callable] = None,
    ) -> "GAN":
        Xt = self._check_tensor(
            X,
        )

        self._train(
            Xt,
            fake_labels_generator=fake_labels_generator,
            true_labels_generator=true_labels_generator,
        )

        return self

    def generate(self, count: int) -> np.ndarray:
        return self(count).cpu().numpy()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, count: int) -> torch.Tensor:
        fixed_noise = torch.randn(count, self.n_units_latent, device=DEVICE)
        with torch.no_grad():
            return self.generator(fixed_noise).detach().cpu()

    def train_epoch_generator(
        self,
        fake_labels_generator: Optional[Callable] = None,
        true_labels_generator: Optional[Callable] = None,
    ) -> float:
        # Update G network: maximize log(D(G(z)))
        if fake_labels_generator is None:
            fake_labels_generator = self.fake_labels_generator
        if true_labels_generator is None:
            true_labels_generator = self.true_labels_generator

        self.generator.optimizer.zero_grad()

        noise = torch.randn(self.batch_size, self.n_units_latent, device=DEVICE)
        fake = self.generator(noise)

        label = self.true_labels_generator(
            fake
        ).squeeze()  # All generated items look real for the generator

        output = self.discriminator(fake).squeeze().float()
        # Calculate G's loss based on this output
        errG = self.criterion(output, label)
        # Calculate gradients for G
        errG.backward()

        # Update G
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.clipping_value)
        self.generator.optimizer.step()

        # Return loss
        return errG.item()

    def train_epoch_discriminator(
        self,
        X: torch.Tensor,
        fake_labels_generator: Optional[Callable] = None,
        true_labels_generator: Optional[Callable] = None,
    ) -> float:
        # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        if fake_labels_generator is None:
            fake_labels_generator = self.fake_labels_generator

        if true_labels_generator is None:
            true_labels_generator = self.true_labels_generator

        errors = []
        for epoch in range(self.discriminator_n_iter):
            self.discriminator.zero_grad()

            # Train with all-real batch
            real_X, label = X.to(DEVICE), true_labels_generator(X).squeeze()
            output = self.discriminator(real_X).squeeze().float()
            errD_real = self.criterion(output, label)
            errD_real.backward()

            # Train with all-fake batch
            noise = torch.randn(self.batch_size, self.n_units_latent, device=DEVICE)
            fake = self.generator(noise)
            label = fake_labels_generator(fake).squeeze().float()

            output = self.discriminator(fake.detach()).squeeze()
            errD_fake = self.criterion(output, label)
            errD_fake.backward()

            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake

            # Update D
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), self.clipping_value
            )
            self.discriminator.optimizer.step()

            errors.append(errD.item())

        return np.mean(errors)

    def train_epoch(
        self,
        loader: DataLoader,
        fake_labels_generator: Optional[Callable] = None,
        true_labels_generator: Optional[Callable] = None,
    ) -> Tuple[float, float]:
        G_losses = []
        D_losses = []

        for i, data in enumerate(loader):
            D_losses.append(
                self.train_epoch_discriminator(
                    data[0],
                    fake_labels_generator=fake_labels_generator,
                    true_labels_generator=true_labels_generator,
                )
            )
            G_losses.append(
                self.train_epoch_generator(
                    fake_labels_generator=fake_labels_generator,
                    true_labels_generator=true_labels_generator,
                )
            )

        return np.mean(G_losses), np.mean(D_losses)

    def dataloader(self, X: torch.Tensor) -> DataLoader:
        dataset = TensorDataset(X)
        return DataLoader(dataset, batch_size=self.batch_size, pin_memory=False)

    def _train(
        self,
        X: torch.Tensor,
        fake_labels_generator: Optional[Callable] = None,
        true_labels_generator: Optional[Callable] = None,
    ) -> "GAN":
        X = self._check_tensor(X).float()

        # Load Dataset
        loader = self.dataloader(X)

        # Train loop
        for i in range(self.generator_n_iter):
            g_loss, d_loss = self.train_epoch(
                loader,
                fake_labels_generator=fake_labels_generator,
                true_labels_generator=true_labels_generator,
            )
            # Check how the generator is doing by saving G's output on fixed_noise
            if i % self.n_iter_print == 0:
                log.info(
                    f"[{i}/{self.generator_n_iter}]\tLoss_D: {d_loss}\tLoss_G: {g_loss}"
                )

        return self

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(DEVICE)
        else:
            return torch.from_numpy(np.asarray(X)).to(DEVICE)
