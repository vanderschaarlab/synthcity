# third party
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torchvision import datasets, transforms

# synthcity absolute
from synthcity.plugins.core.models.image_gan import ImageGAN
from synthcity.utils.constants import DEVICE

IMG_SIZE = 28
data_transform = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ]
)
# Load MNIST dataset as tensors
batch_size = 128
dataset = datasets.MNIST(".", download=True, transform=data_transform)
dataset = Subset(dataset, np.arange(len(dataset))[:100])


class Generator(nn.Module):
    def __init__(
        self, no_of_channels: int = 1, noise_dim: int = 100, gen_dim: int = 32
    ) -> None:
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, gen_dim * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gen_dim * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_dim * 4, gen_dim * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(gen_dim * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_dim * 2, gen_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_dim, no_of_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.network(X)
        return output


class Discriminator(nn.Module):
    def __init__(self, no_of_channels: int = 1, disc_dim: int = 32) -> None:
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(no_of_channels, disc_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_dim, disc_dim * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(disc_dim * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_dim * 2, disc_dim * 4, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(disc_dim * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_dim * 4, 1, 4, 1, 0, bias=False),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.network(X)
        return output


def test_network_config() -> None:
    noise_dim = 123
    out_dim = 100
    net = ImageGAN(
        image_generator=Generator(noise_dim=noise_dim, gen_dim=out_dim),
        image_discriminator=Discriminator(disc_dim=out_dim),
        n_units_latent=noise_dim,
        # Generator
        generator_n_iter=1001,
        generator_lr=1e-3,
        generator_weight_decay=1e-3,
        # Discriminator
        discriminator_n_iter=1002,
        discriminator_lr=1e-3,
        discriminator_weight_decay=1e-3,
        # Training
        batch_size=64,
        n_iter_print=100,
        random_state=77,
        n_iter_min=100,
        clipping_value=1,
        lambda_gradient_penalty=2,
        lambda_identifiability_penalty=3,
    )

    assert isinstance(net.generator, Generator)
    assert isinstance(net.discriminator, Discriminator)
    assert net.batch_size == 64
    assert net.generator_n_iter == 1001
    assert net.discriminator_n_iter == 1002
    assert net.random_state == 77
    assert net.lambda_gradient_penalty == 2
    assert net.lambda_identifiability_penalty == 3


@pytest.mark.parametrize("n_iter", [10])
@pytest.mark.parametrize("lr", [1e-3, 3e-4])
def test_basic_network(
    n_iter: int,
    lr: float,
) -> None:
    noise_dim = 123
    out_dim = 100

    net = ImageGAN(
        image_generator=Generator(noise_dim=noise_dim, gen_dim=out_dim),
        image_discriminator=Discriminator(disc_dim=out_dim),
        n_units_latent=noise_dim,
        generator_n_iter=n_iter,
        discriminator_n_iter=n_iter,
        generator_lr=lr,
        discriminator_lr=lr,
    )

    assert net.generator_n_iter == n_iter
    assert net.discriminator_n_iter == n_iter


@pytest.mark.parametrize("generator_extra_penalties", [[], ["identifiability_penalty"]])
def test_gan_generation(generator_extra_penalties: list) -> None:
    noise_dim = 123
    out_dim = 100

    model = ImageGAN(
        image_generator=Generator(noise_dim=noise_dim, gen_dim=out_dim).to(DEVICE),
        image_discriminator=Discriminator(disc_dim=out_dim).to(DEVICE),
        n_units_latent=noise_dim,
        n_channels=1,
        generator_n_iter=10,
        generator_extra_penalties=generator_extra_penalties,
    )
    model.fit(dataset)

    generated = model.generate(10)

    assert generated.shape == (10, 1, IMG_SIZE, IMG_SIZE)
