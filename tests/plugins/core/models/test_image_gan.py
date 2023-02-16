# third party
import numpy as np
import pytest
from monai.networks.nets import Discriminator, Generator
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


def generator(latent_size: int) -> Generator:
    # upsampling

    return Generator(
        latent_shape=latent_size,
        start_shape=(64, 7, 7),
        channels=[128, 64, 1],
        strides=[2, 2, 1],
        kernel_size=3,
        bias=False,
    ).to(DEVICE)


def discriminator() -> Discriminator:
    return Discriminator(
        in_shape=(1, IMG_SIZE, IMG_SIZE),
        channels=[32, 64, 128, 1],
        strides=[2, 2, 2, 2],
        kernel_size=3,
        last_act=None,
        bias=False,
    ).to(DEVICE)


def test_network_config() -> None:
    noise_dim = 123
    net = ImageGAN(
        image_generator=generator(noise_dim),
        image_discriminator=discriminator(),
        n_units_latent=noise_dim,
        n_channels=1,
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

    net = ImageGAN(
        image_generator=generator(noise_dim),
        image_discriminator=discriminator(),
        n_units_latent=noise_dim,
        n_channels=1,
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

    model = ImageGAN(
        image_generator=generator(noise_dim).to(DEVICE),
        image_discriminator=discriminator().to(DEVICE),
        n_units_latent=noise_dim,
        n_channels=1,
        generator_n_iter=10,
        generator_extra_penalties=generator_extra_penalties,
    )
    model.fit(dataset)

    generated = model.generate(10)

    assert generated.shape == (10, 1, IMG_SIZE, IMG_SIZE)
