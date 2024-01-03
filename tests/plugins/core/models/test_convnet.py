# third party
import numpy as np
import pytest
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

# synthcity absolute
from synthcity.plugins.core.models.convnet import (
    map_nonlin,
    suggest_image_classifier_arch,
    suggest_image_generator_discriminator_arch,
)
from synthcity.utils.constants import DEVICE


@pytest.mark.parametrize("nonlin", ["relu", "elu", "prelu", "leaky_relu"])
def test_get_nonlin(nonlin: str) -> None:
    assert map_nonlin(nonlin) is not None


@pytest.mark.parametrize("n_channels", [1, 3])
@pytest.mark.parametrize("height", [32, 64, 128])
def test_suggest_gan(n_channels: int, height: int) -> None:
    n_units_latent = 100
    gen, disc = suggest_image_generator_discriminator_arch(
        n_units_latent=n_units_latent,
        n_channels=n_channels,
        height=height,
        width=height,
    )

    dummy_noise = torch.rand((10, n_units_latent, n_channels, 1), device=DEVICE)
    gen(dummy_noise)

    dummy_in = torch.rand((10, n_channels, height, height), device=DEVICE)
    disc(dummy_in)


@pytest.mark.parametrize("n_channels", [1, 3])
@pytest.mark.parametrize("height", [32, 64, 128])
def test_suggest_clf(n_channels: int, height: int) -> None:
    classes = 13
    clf = suggest_image_classifier_arch(
        n_channels=n_channels,
        height=height,
        width=height,
        classes=classes,
    )

    dummy_input = torch.rand((10, n_channels, height, height))
    out = clf(dummy_input)

    assert out.shape == (10, classes)


def test_train_clf() -> None:
    IMG_SIZE = 32
    data_transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )
    dataset = datasets.MNIST(".", download=True, transform=data_transform)
    dataset = Subset(dataset, np.arange(len(dataset))[:100])

    classes = 10

    clf = suggest_image_classifier_arch(
        n_channels=1,
        height=IMG_SIZE,
        width=IMG_SIZE,
        classes=classes,
        n_iter=100,
        n_iter_print=10,
        batch_size=40,
    )

    clf.fit(dataset)

    test_X, test_y = next(iter(dataset))

    print(clf.predict(test_X), test_y)
