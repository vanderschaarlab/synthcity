# stdlib
from typing import Any, Tuple

# third party
import numpy as np
import pytest
from sklearn.datasets import load_digits, load_iris
from sklearn.preprocessing import MinMaxScaler

# synthcity absolute
from synthcity.metrics.weighted_metrics import WeightedMetrics
from synthcity.plugins.core.models.gan import GAN


def test_network_config() -> None:
    net = GAN(
        n_features=10,
        n_units_latent=2,
        # Generator
        generator_n_layers_hidden=2,
        generator_n_units_hidden=100,
        generator_nonlin="elu",
        generator_nonlin_out=[("sigmoid", 10)],
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
        lambda_gradient_penalty=2,
        lambda_identifiability_penalty=3,
    )

    assert len(net.generator.model) == 4
    assert len(net.discriminator.model) == 5
    assert net.batch_size == 64
    assert net.generator_n_iter == 1001
    assert net.discriminator_n_iter == 1002
    assert net.random_state == 77
    assert net.lambda_gradient_penalty == 2
    assert net.lambda_identifiability_penalty == 3


@pytest.mark.parametrize("nonlin", ["relu", "elu", "leaky_relu"])
@pytest.mark.parametrize("n_iter", [10])
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


@pytest.mark.parametrize("generator_extra_penalties", [[], ["identifiability_penalty"]])
def test_gan_generation(generator_extra_penalties: list) -> None:
    X, _ = load_digits(return_X_y=True)
    X = MinMaxScaler().fit_transform(X)

    model = GAN(
        n_features=X.shape[1],
        n_units_latent=50,
        generator_n_iter=10,
        generator_extra_penalties=generator_extra_penalties,
    )
    model.fit(X)

    generated = model.generate(10)

    assert generated.shape == (10, X.shape[1])


def test_gan_conditional() -> None:
    X, y = load_iris(return_X_y=True)
    X = MinMaxScaler().fit_transform(X)

    model = GAN(
        n_features=X.shape[1],
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
    X, _ = load_iris(return_X_y=True)
    X = MinMaxScaler().fit_transform(X)

    model = GAN(
        n_features=X.shape[1],
        n_units_latent=50,
        generator_n_iter=50,
        n_iter_print=10,
        dp_enabled=True,
    )
    model.fit(X)

    generated = model.generate(10)

    assert generated.shape == (10, X.shape[1])


@pytest.mark.parametrize(
    "patience_metric",
    [
        ("detection", "detection_mlp"),
        ("performance", "xgb"),
    ],
)
def test_gan_generation_with_early_stopping(patience_metric: Tuple[str, str]) -> None:
    X, _ = load_iris(return_X_y=True)
    X = MinMaxScaler().fit_transform(X)
    actual_iter = 0

    def _tracker(*args: Any, **kwargs: Any) -> float:
        nonlocal actual_iter
        actual_iter += 1
        return 0

    model = GAN(
        n_features=X.shape[1],
        n_units_latent=50,
        generator_n_iter=1000,
        n_iter_print=20,
        patience=2,
        batch_size=len(X),
        patience_metric=WeightedMetrics(metrics=[patience_metric], weights=[1]),
        generator_extra_penalty_cbks=[_tracker],
    )
    model.fit(X)

    generated = model.generate(10)

    assert generated.shape == (10, X.shape[1])

    assert actual_iter < 1000
