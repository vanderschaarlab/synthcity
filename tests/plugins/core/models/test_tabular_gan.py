# third party
import numpy as np
import pytest
from helpers import get_airfoil_dataset
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.plugins.core.models.tabular_gan import TabularGAN


def test_network_config() -> None:
    X, _ = load_iris(return_X_y=True, as_frame=True)
    net = TabularGAN(
        X,
        n_units_latent=2,
        # Generator
        generator_n_layers_hidden=2,
        generator_n_units_hidden=100,
        generator_nonlin="elu",
        generator_nonlin_out_discrete="sigmoid",
        generator_nonlin_out_continuous="tanh",
        generator_n_iter=1001,
        generator_batch_norm=False,
        generator_dropout=0,
        generator_lr=1e-3,
        generator_weight_decay=1e-3,
        # Discriminator
        discriminator_n_layers_hidden=3,
        discriminator_n_units_hidden=100,
        discriminator_nonlin="elu",
        discriminator_n_iter=1002,
        discriminator_batch_norm=False,
        discriminator_dropout=0,
        discriminator_lr=1e-3,
        discriminator_weight_decay=1e-3,
        # Training
        batch_size=64,
        n_iter_print=100,
        random_state=77,
        n_iter_min=100,
        clipping_value=1,
        encoder_max_clusters=5,
    )

    assert len(net.model.generator.model) == 4
    assert len(net.model.discriminator.model) == 5
    assert net.model.batch_size == 64
    assert net.model.generator_n_iter == 1001
    assert net.model.discriminator_n_iter == 1002
    assert net.model.random_state == 77
    assert net.encoder is not None


@pytest.mark.parametrize("nonlin", ["relu", "elu", "leaky_relu"])
@pytest.mark.parametrize("dropout", [0, 0.5, 0.2])
@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize("lr", [1e-3, 3e-4])
def test_basic_network(
    nonlin: str,
    dropout: float,
    batch_norm: bool,
    lr: float,
) -> None:
    X, _ = load_iris(return_X_y=True, as_frame=True)
    net = TabularGAN(
        X,
        n_units_latent=2,
        generator_n_iter=3,
        discriminator_n_iter=3,
        generator_dropout=dropout,
        discriminator_dropout=dropout,
        generator_nonlin=nonlin,
        discriminator_nonlin=nonlin,
        generator_batch_norm=batch_norm,
        discriminator_batch_norm=batch_norm,
        generator_n_layers_hidden=2,
        discriminator_n_layers_hidden=2,
        generator_lr=lr,
        discriminator_lr=lr,
        encoder_max_clusters=5,
    )

    assert net.model.generator_n_iter == 3
    assert net.model.discriminator_n_iter == 3
    assert net.model.generator.lr == lr
    assert net.model.discriminator.lr == lr


def test_gan_classification() -> None:
    X = get_airfoil_dataset()

    model = TabularGAN(
        X,
        n_units_latent=50,
        generator_n_iter=10,
        encoder_max_clusters=5,
    )
    model.fit(X)

    generated = model.generate(10)

    assert (X.columns == generated.columns).all()
    assert generated.shape == (10, X.shape[1])


def test_gan_conditional() -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)

    model = TabularGAN(
        X,
        n_units_latent=50,
        n_units_conditional=1,
        generator_n_iter=10,
    )
    model.fit(X, cond=y)

    generated = model.generate(10)
    assert generated.shape == (10, X.shape[1])

    generated = model.generate(5, np.ones(5))
    assert generated.shape == (5, X.shape[1])


def test_gan_generation_with_dp() -> None:
    X = get_airfoil_dataset()

    model = TabularGAN(
        X,
        n_units_latent=50,
        generator_n_iter=10,
        encoder_max_clusters=5,
        dp_enabled=True,
    )
    model.fit(X)

    generated = model.generate(10)

    assert (X.columns == generated.columns).all()
    assert generated.shape == (10, X.shape[1])
