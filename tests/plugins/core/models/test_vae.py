# third party
import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler

# synthcity absolute
from synthcity.plugins.core.models.vae import VAE


def test_network_config() -> None:
    net = VAE(
        n_features=10,
        n_units_embedding=2,
        # decoder
        decoder_n_layers_hidden=2,
        decoder_n_units_hidden=100,
        decoder_nonlin="elu",
        decoder_nonlin_out=[("sigmoid", 10)],
        decoder_batch_norm=False,
        decoder_dropout=0,
        # encoder
        encoder_n_layers_hidden=3,
        encoder_n_units_hidden=100,
        encoder_nonlin="elu",
        encoder_batch_norm=False,
        encoder_dropout=0,
        # Training
        n_iter=1001,
        lr=1e-3,
        weight_decay=1e-3,
        batch_size=64,
        n_iter_print=100,
        random_state=77,
        clipping_value=1,
    )

    assert len(net.decoder.model) == 4
    assert len(net.encoder.shared) == 3
    assert net.batch_size == 64
    assert net.n_iter == 1001
    assert net.lr == 1e-3
    assert net.weight_decay == 1e-3


@pytest.mark.parametrize("nonlin", ["relu", "elu", "leaky_relu"])
@pytest.mark.parametrize("dropout", [0, 0.5, 0.2])
@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize("lr", [1e-3, 3e-4])
@pytest.mark.parametrize("hidden", [2, 3])
def test_basic_network(
    nonlin: str,
    dropout: float,
    batch_norm: bool,
    lr: float,
    hidden: int,
) -> None:
    net = VAE(
        n_features=10,
        n_units_embedding=2,
        n_iter=10,
        lr=lr,
        decoder_dropout=dropout,
        encoder_dropout=dropout,
        decoder_nonlin=nonlin,
        encoder_nonlin=nonlin,
        decoder_batch_norm=batch_norm,
        encoder_batch_norm=batch_norm,
        decoder_n_layers_hidden=hidden,
        encoder_n_layers_hidden=hidden,
    )

    assert net.n_iter == 10
    assert net.lr == lr
    assert len(net.decoder.model) == hidden + 1
    assert len(net.encoder.shared) == hidden


@pytest.mark.parametrize("loss_strategy", ["standard", "robust_divergence"])
def test_vae_classification(loss_strategy: str) -> None:
    X, _ = load_digits(return_X_y=True)
    X = MinMaxScaler().fit_transform(X)

    model = VAE(
        n_features=X.shape[1],
        n_units_embedding=50,
        n_iter=10,
        loss_strategy=loss_strategy,
    )
    model.fit(X)

    generated = model.generate(10)

    assert generated.shape == (10, X.shape[1])


def test_vae_conditional() -> None:
    X, y = load_digits(return_X_y=True)
    X = MinMaxScaler().fit_transform(X)

    model = VAE(
        n_features=X.shape[1],
        n_units_embedding=50,
        n_units_conditional=1,
        n_iter=10,
    )
    model.fit(X, cond=y)

    generated = model.generate(10)
    assert generated.shape == (10, X.shape[1])

    generated = model.generate(5, np.ones(5))
    assert generated.shape == (5, X.shape[1])
