# third party
import pytest
from sklearn.datasets import load_digits

# synthcity absolute
from synthcity.plugins.models.gan import GAN


def test_network_config() -> None:
    net = GAN(
        n_features=10,
        n_units_latent=2,
        # Generator
        generator_n_layers_hidden=2,
        generator_n_units_hidden=100,
        generator_nonlin="elu",
        generator_nonlin_out="sigmoid",
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
        seed=77,
        n_iter_min=100,
        clipping_value=1,
    )

    assert len(net.generator.model) == 6
    assert len(net.discriminator.model) == 8
    assert net.batch_size == 64
    assert net.generator_n_iter == 1001
    assert net.discriminator_n_iter == 1002
    assert net.seed == 77


@pytest.mark.parametrize("nonlin", ["relu", "elu", "leaky_relu"])
@pytest.mark.parametrize("n_iter", [10, 50, 100])
@pytest.mark.parametrize("dropout", [0, 0.5, 0.2])
@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize("lr", [1e-3, 3e-4])
def test_basic_network(
    nonlin: str,
    n_iter: int,
    dropout: float,
    batch_norm: bool,
    lr: float,
) -> None:
    net = GAN(
        n_features=10,
        n_units_latent=2,
        generator_n_iter=n_iter,
        discriminator_n_iter=n_iter,
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
    )

    assert net.generator_n_iter == n_iter
    assert net.discriminator_n_iter == n_iter
    assert net.generator.lr == lr
    assert net.discriminator.lr == lr


def test_gan_classification() -> None:
    X, _ = load_digits(return_X_y=True)
    model = GAN(
        n_features=X.shape[1],
        n_units_latent=50,
    )
    model.fit(X)

    generated = model.generate(10)

    assert generated.shape == (10, X.shape[1])
