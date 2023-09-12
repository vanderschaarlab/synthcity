# stdlib
from typing import Any, Callable, List, Optional, Tuple

# third party
import matplotlib.pyplot as plt
import numpy as np
import torch
from opacus import PrivacyEngine
from pydantic import validate_arguments
from torch import nn
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm

# synthcity absolute
import synthcity.logger as log
from synthcity.metrics.weighted_metrics import WeightedMetrics
from synthcity.plugins.core.dataloader import ImageDataLoader
from synthcity.plugins.core.dataset import ConditionalDataset, FlexibleDataset
from synthcity.utils.constants import DEVICE
from synthcity.utils.reproducibility import clear_cache, enable_reproducible_results


def display_imgs(imgs: List[np.ndarray], title: Optional[str] = None) -> None:
    for i in range(len(imgs)):
        plt.subplot(1, len(imgs), i + 1)
        plt.tight_layout()
        imgs[i] = imgs[i] / 2 + 0.5  # unnormalize
        plt.imshow(np.transpose(imgs[i].cpu().numpy(), (1, 2, 0)))

    if title is not None:
        plt.title(title)
    plt.show()


def weights_init(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(m.bias, val=0)


class ImageGAN(nn.Module):
    """
    .. inheritance-diagram:: synthcity.plugins.core.models.image_gan.ImageGAN
        :parts: 1

    Basic GAN implementation.

    Args:
        image_generator: nn.Module
            Generator model
        image_discriminator: nn.Module
            Discriminator model
        n_units_latent: int
            Number of latent units
        n_channels: int
            Number of channels in the image
        generator_n_iter: int
            Maximum number of iterations in the Generator.
        generator_lr: float = 2e-4
            Generator learning rate, used by the Adam optimizer
        generator_weight_decay: float = 1e-3
            Generator weight decay, used by the Adam optimizer
        generator_opt_betas: tuple = (0.9, 0.999)
            Generator initial decay rates, used by the Adam Optimizer
        generator_extra_penalties: list
            Additional penalties for the generator. Values: "identifiability_penalty"
        generator_extra_penalty_cbks: List[Callable]
            Additional loss callabacks for the generator. Used by the TabularGAN for the conditional loss
        discriminator_n_iter: int
            Maximum number of iterations in the discriminator.
        discriminator_lr: float
            Discriminator learning rate, used by the Adam optimizer
        discriminator_weight_decay: float
            Discriminator weight decay, used by the Adam optimizer
        discriminator_opt_betas: tuple
            Initial weight decays for the Adam optimizer
        batch_size: int
            Batch size
        random_state: int
            random_state used
        clipping_value: int, default 0
            Gradients clipping value. Zero disables the feature
        lambda_gradient_penalty: float = 10
            Weight for the gradient penalty
        lambda_identifiability_penalty: float = 0.1
            Weight for the identifiability penalty, if enabled
        dataloader_sampler: Optional[sampler.Sampler]
            Optional sampler for the dataloader, useful for conditional sampling
        device: Any = DEVICE
            CUDA/CPU
        # early stopping
        n_iter_print: int
            Number of iterations after which to print updates and check the validation loss.
        n_iter_min: int
            Minimum number of iterations to go through before starting early stopping
        patience: int
            Max number of iterations without any improvement before early stopping is trigged.
        patience_metric: Optional[WeightedMetrics]
            If not None, the metric is used for evaluation the criterion for early stopping.
        # privacy settings
        dp_enabled: bool
            Train the discriminator with Differential Privacy guarantees
        dp_delta: Optional[float]
            Optional DP delta: the probability of information accidentally being leaked. Usually 1 / len(dataset)
        dp_epsilon: float = 3
            DP epsilon: privacy budget, which is a measure of the amount of privacy that is preserved by a given algorithm. Epsilon is a number that represents the maximum amount of information that an adversary can learn about an individual from the output of a differentially private algorithm. The smaller the value of epsilon, the more private the algorithm is. For example, an algorithm with an epsilon of 0.1 preserves more privacy than an algorithm with an epsilon of 1.0.
        dp_max_grad_norm: float
            max grad norm used for gradient clipping
        dp_secure_mode: bool = False,
             if True uses noise generation approach robust to floating point arithmetic attacks.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        image_generator: nn.Module,
        image_discriminator: nn.Module,
        n_units_latent: int,
        n_channels: int,
        # generator
        generator_n_iter: int = 500,
        generator_lr: float = 2e-4,
        generator_weight_decay: float = 1e-3,
        generator_opt_betas: tuple = (0.9, 0.999),
        generator_extra_penalties: list = [],  # "identifiability_penalty"
        generator_extra_penalty_cbks: List[Callable] = [],
        # discriminator
        discriminator_n_iter: int = 1,
        discriminator_lr: float = 2e-4,
        discriminator_weight_decay: float = 1e-3,
        discriminator_opt_betas: tuple = (0.9, 0.999),
        # training
        batch_size: int = 100,
        random_state: int = 0,
        clipping_value: int = 1,
        lambda_gradient_penalty: float = 10,
        lambda_identifiability_penalty: float = 0.1,
        device: Any = DEVICE,
        n_iter_min: int = 100,
        n_iter_print: int = 1,
        plot_progress: int = False,
        patience: int = 20,
        patience_metric: Optional[WeightedMetrics] = None,
        dataloader_sampler: Optional[sampler.Sampler] = None,
        # privacy settings
        dp_enabled: bool = False,
        dp_delta: Optional[float] = None,
        dp_epsilon: float = 3,
        dp_max_grad_norm: float = 2,
        dp_secure_mode: bool = False,
    ) -> None:
        super(ImageGAN, self).__init__()

        extra_penalty_list = ["identifiability_penalty"]
        for penalty in generator_extra_penalties:
            if penalty not in extra_penalty_list:
                raise ValueError(f"Unsupported generator penalty {penalty}")

        log.info(f"Training ImageGAN on device {device}.")
        self.device = device
        self.generator_extra_penalties = generator_extra_penalties
        self.generator_extra_penalty_cbks = generator_extra_penalty_cbks

        self.generator = image_generator.apply(weights_init)
        self.discriminator = image_discriminator.apply(weights_init)

        self.n_units_latent = n_units_latent
        self.n_channels = n_channels
        self.plot_progress = plot_progress

        # training
        self.generator_n_iter = generator_n_iter
        self.generator_lr = generator_lr
        self.generator_weight_decay = generator_weight_decay
        self.generator_opt_betas = generator_opt_betas

        self.discriminator_n_iter = discriminator_n_iter
        self.discriminator_lr = discriminator_lr
        self.discriminator_weight_decay = discriminator_weight_decay
        self.discriminator_opt_betas = discriminator_opt_betas

        self.n_iter_print = n_iter_print
        self.n_iter_min = n_iter_min
        self.patience = patience
        self.patience_metric = patience_metric
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

        # privacy
        self.dp_enabled = dp_enabled
        self.dp_delta = dp_delta
        self.dp_epsilon = dp_epsilon
        self.dp_max_grad_norm = dp_max_grad_norm
        self.dp_secure_mode = dp_secure_mode

    def _get_noise(self, n_samples: int) -> torch.Tensor:
        """
        Generate noise vectors from the random normal distribution with dimensions (n_samples, noise_dim),
        where
            n_samples: the number of samples to generate based on  batch_size
        """

        return torch.randn(
            n_samples, self.n_units_latent, self.n_channels, 1, device=self.device
        )

    def fit(
        self,
        X: FlexibleDataset,
        cond: Optional[torch.Tensor] = None,
        fake_labels_generator: Optional[Callable] = None,
        true_labels_generator: Optional[Callable] = None,
    ) -> "ImageGAN":
        clear_cache()

        self.with_conditional = False
        if cond is not None:
            cond = self._check_tensor(cond)
            self.with_conditional = True

        self._train(
            X,
            cond=cond,
            fake_labels_generator=fake_labels_generator,
            true_labels_generator=true_labels_generator,
        )

        return self

    def generate(
        self,
        count: int,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        clear_cache()
        self.generator.eval()
        with torch.no_grad():
            return self(count, cond=cond).detach()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(
        self,
        count: int,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fixed_noise = self._get_noise(count)

        if cond is not None:
            cond = self._check_tensor(cond)

        return self.generator(fixed_noise, cond=cond)

    def dataloader(
        self, dataset: torch.utils.data.Dataset, cond: Optional[torch.Tensor] = None
    ) -> DataLoader:
        if cond is None:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=self.dataloader_sampler,
                pin_memory=False,
                shuffle=True,
            )

        cond_dataset = ConditionalDataset(dataset, cond)
        return DataLoader(
            cond_dataset,
            batch_size=self.batch_size,
            sampler=self.dataloader_sampler,
            pin_memory=False,
            shuffle=True,
        )

    def _train_epoch_generator(
        self,
        X: torch.Tensor,
        fake_labels_generator: Callable,
        true_labels_generator: Callable,
        cond: Optional[torch.Tensor] = None,
    ) -> float:
        # Update the G network
        self.generator.train()

        real_X = X.to(self.device)
        batch_size = len(real_X)

        noise = self._get_noise(batch_size)

        fake = self.generator(noise, cond=cond)

        output = self.discriminator(fake, cond=cond).squeeze().float()
        # Calculate G's loss based on this output
        errG = -torch.mean(output)
        for extra_loss in self.generator_extra_penalty_cbks:
            errG += extra_loss(
                real_X,
                fake,
            )

        errG += self._extra_penalties(
            self.generator_extra_penalties,
            real_samples=real_X,
            fake_samples=fake,
            batch_size=batch_size,
        )

        # Calculate gradients for G
        self.generator_optimizer.zero_grad()
        errG.backward()

        # Update G
        if self.clipping_value > 0:
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), self.clipping_value
            )
        self.generator_optimizer.step()

        if torch.isnan(errG):
            raise RuntimeError("NaNs detected in the generator loss")

        # Return loss
        return errG.item()

    def _train_epoch_discriminator(
        self,
        X: torch.Tensor,
        fake_labels_generator: Callable,
        true_labels_generator: Callable,
        cond: Optional[torch.Tensor] = None,
    ) -> float:
        # Update the D network
        self.discriminator.train()

        errors = []

        batch_size = min(self.batch_size, len(X))

        for epoch in range(self.discriminator_n_iter):
            # Train with all-real batch
            real_X = X.to(self.device)

            real_labels = true_labels_generator(X).to(self.device).squeeze()
            real_output = self.discriminator(real_X, cond=cond).squeeze().float()

            # Train with all-fake batch
            noise = self._get_noise(batch_size)

            fake = self.generator(noise, cond=cond)

            fake_labels = fake_labels_generator(fake).to(self.device).squeeze().float()
            fake_output = self.discriminator(fake.detach(), cond=cond).squeeze()

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
                cond=cond,
            )
            errD = -errD_real + errD_fake

            self.discriminator_optimizer.zero_grad()
            # TODO: investigate DP support for image generation. The current version is not functional
            if self.dp_enabled:
                # Adversarial loss
                # 1. split fwd-bkwd on fake and real images into two explicit blocks.
                # 2. no need to compute per_sample_gardients on fake data, disable hooks.
                # 3. re-enable hooks to obtain per_sample_gardients for real data.
                # fake fwd-bkwd
                self.discriminator.disable_hooks()
                penalty.backward(retain_graph=True)
                errD_fake.backward(retain_graph=True)

                self.discriminator.enable_hooks()
                errD_real.backward()  # HACK: calling bkwd without zero_grad() accumulates param gradients
            else:
                errD += penalty
                errD.backward()

            # Update D
            if self.clipping_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), self.clipping_value
                )
            self.discriminator_optimizer.step()

            errors.append(errD.item())

        if np.isnan(np.mean(errors)):
            raise RuntimeError("NaNs detected in the discriminator loss")

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
            if self.with_conditional:
                X, cond = data
            else:
                X = data[0]

            D_losses.append(
                self._train_epoch_discriminator(
                    X,
                    fake_labels_generator=fake_labels_generator,
                    true_labels_generator=true_labels_generator,
                    cond=cond,
                )
            )
            G_losses.append(
                self._train_epoch_generator(
                    X,
                    fake_labels_generator=fake_labels_generator,
                    true_labels_generator=true_labels_generator,
                    cond=cond,
                )
            )

        return np.mean(G_losses), np.mean(D_losses)

    def _init_patience_score(self) -> float:
        if self.patience_metric is None:
            return 0

        if self.patience_metric.direction() == "minimize":
            return np.inf
        else:
            return -np.inf

    def _evaluate_patience_metric(
        self,
        X: torch.Tensor,
        cond: Optional[torch.Tensor],
        prev_score: float,
        patience: int,
    ) -> Tuple[float, int, bool]:
        save = False
        if self.patience_metric is None:
            return prev_score, patience, save

        X_syn = self.generate(len(X), cond=cond)
        new_score = self.patience_metric.evaluate(
            ImageDataLoader(ConditionalDataset(X)),
            ImageDataLoader(ConditionalDataset(X_syn)),
        )
        score = prev_score
        if self.patience_metric.direction() == "minimize":
            if new_score >= prev_score:
                patience += 1
            else:
                patience = 0
                score = new_score
                save = True
        else:
            if new_score <= prev_score:
                patience += 1
            else:
                patience = 0
                score = new_score
                save = True

        return score, patience, save

    def _train_test_split(
        self, X: FlexibleDataset, cond: Optional[torch.Tensor] = None
    ) -> Tuple:
        if self.patience_metric is None:
            return X, cond, None, None

        if self.dataloader_sampler is not None:
            train_idx, test_idx = self.dataloader_sampler.train_test()
        else:
            total = np.arange(0, len(X))
            np.random.shuffle(total)
            split = int(len(total) * 0.8)
            train_idx, test_idx = total[:split], total[split:]

        X_train, X_val = X.filter_indices(train_idx), X.filter_indices(test_idx)
        cond_train, cond_val = None, None
        if cond is not None:
            cond_train, cond_val = cond[train_idx], cond[test_idx]
        return X_train, cond_train, X_val, cond_val

    def _train(
        self,
        X: FlexibleDataset,
        cond: Optional[torch.Tensor] = None,
        fake_labels_generator: Optional[Callable] = None,
        true_labels_generator: Optional[Callable] = None,
    ) -> "ImageGAN":
        self.train()

        X, cond, X_val, cond_val = self._train_test_split(X, cond)

        # Load Dataset
        loader = self.dataloader(X, cond)

        # Create the optimizers
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.generator_lr,
            weight_decay=self.generator_weight_decay,
            betas=self.generator_opt_betas,
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.discriminator_lr,
            weight_decay=self.discriminator_weight_decay,
            betas=self.discriminator_opt_betas,
        )

        # Privacy
        if self.dp_enabled:
            if self.dp_delta is None:
                self.dp_delta = 1 / len(X)

            privacy_engine = PrivacyEngine(secure_mode=self.dp_secure_mode)

            (
                self.discriminator,
                self.discriminator_optimizer,
                loader,
            ) = privacy_engine.make_private_with_epsilon(
                module=self.discriminator,
                optimizer=self.discriminator_optimizer,
                data_loader=loader,
                epochs=self.generator_n_iter,
                target_epsilon=self.dp_epsilon,
                target_delta=self.dp_delta,
                max_grad_norm=self.dp_max_grad_norm,
                poisson_sampling=False,
            )

        # Train loop
        patience_score = self._init_patience_score()
        patience = 0
        best_state_dict = None

        for i in tqdm(range(self.generator_n_iter)):
            g_loss, d_loss = self._train_epoch(
                loader,
                fake_labels_generator=fake_labels_generator,
                true_labels_generator=true_labels_generator,
            )
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i + 1) % self.n_iter_print == 0:
                if self.plot_progress:
                    display_imgs(self.generate(5), title="synthetic samples")
                    display_imgs(
                        torch.from_numpy(X_val.numpy()[0])[:5], title="real samples"
                    )

                log.debug(
                    f"[{i}/{self.generator_n_iter}]\tLoss_D: {d_loss}\tLoss_G: {g_loss} Patience score: {patience_score} Patience : {patience}"
                )
                if self.dp_enabled:
                    log.debug(
                        f"[{i}/{self.generator_n_iter}] Privacy budget: epsilon = {privacy_engine.get_epsilon(self.dp_delta)} delta = {self.dp_delta}"
                    )

                if self.patience_metric is not None:
                    patience_score, patience, save = self._evaluate_patience_metric(
                        torch.from_numpy(X_val.numpy()[0]),
                        cond_val,
                        patience_score,
                        patience,
                    )
                    if save:
                        best_state_dict = self.state_dict()

                    if patience >= self.patience and i >= self.n_iter_min:
                        log.debug(f"[{i}/{self.generator_n_iter}] Early stopping")
                        break

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

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
    ) -> torch.Tensor:
        """Calculates additional penalties for the training"""
        err: torch.Tensor = 0
        for penalty in penalties:
            if penalty == "identifiability_penalty":
                err += self._loss_identifiability_penalty(
                    real_samples=real_samples,
                    fake_samples=fake_samples,
                )
            else:
                raise RuntimeError(f"unknown penalty {penalty}")
        return err

    def _loss_gradient_penalty(
        self,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, channel, height, width = real_samples.shape
        # alpha is selected randomly between 0 and 1
        alpha = (
            torch.rand(batch_size, 1, 1, 1)
            .repeat(1, channel, height, width)
            .to(self.device)
        )
        # interpolated image=randomly weighted average between a real and fake image
        interpolatted_samples = (alpha * real_samples) + (1 - alpha) * fake_samples

        # calculate the critic score on the interpolated image
        interpolated_score = self.discriminator(interpolatted_samples, cond=cond)

        # take the gradient of the score wrt to the interpolated image
        gradient = torch.autograd.grad(
            inputs=interpolatted_samples,
            outputs=interpolated_score,
            grad_outputs=torch.ones_like(interpolated_score),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return self.lambda_gradient_penalty * gradient_penalty

    def _loss_identifiability_penalty(
        self,
        real_samples: torch.tensor,
        fake_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the identifiability penalty. Section C in the paper"""
        return (
            -self.lambda_identifiability_penalty
            * (real_samples - fake_samples).square().sum(dim=-1).sqrt().mean()
        )
