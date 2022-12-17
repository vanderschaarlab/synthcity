# stdlib
import sys

# third party
import pytest
from helpers import get_airfoil_dataset
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.plugins.core.models.tabular_vae import TabularVAE


def test_network_config() -> None:
    X, _ = load_iris(return_X_y=True, as_frame=True)
    net = TabularVAE(
        X,
        n_units_embedding=2,
        n_iter=1001,
        lr=1e-3,
        weight_decay=1e-3,
        # decoder
        decoder_n_layers_hidden=2,
        decoder_n_units_hidden=100,
        decoder_nonlin="elu",
        decoder_nonlin_out_discrete="sigmoid",
        decoder_nonlin_out_continuous="tanh",
        decoder_batch_norm=False,
        decoder_dropout=0,
        # encoder
        encoder_n_layers_hidden=3,
        encoder_n_units_hidden=100,
        encoder_nonlin="elu",
        encoder_batch_norm=False,
        encoder_dropout=0,
        # Training
        batch_size=64,
        n_iter_print=100,
        random_state=77,
        clipping_value=1,
        encoder_max_clusters=5,
    )

    assert len(net.model.decoder.model) == 4
    assert len(net.model.encoder.shared) == 3
    assert net.model.batch_size == 64
    assert net.model.n_iter == 1001
    assert net.model.random_state == 77
    assert net.encoder is not None


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
    X, _ = load_iris(return_X_y=True, as_frame=True)
    net = TabularVAE(
        X,
        n_units_embedding=2,
        n_iter=n_iter,
        lr=lr,
        decoder_dropout=dropout,
        encoder_dropout=dropout,
        decoder_nonlin=nonlin,
        encoder_nonlin=nonlin,
        decoder_batch_norm=batch_norm,
        encoder_batch_norm=batch_norm,
        decoder_n_layers_hidden=2,
        encoder_n_layers_hidden=2,
        encoder_max_clusters=5,
    )

    assert net.model.n_iter == n_iter
    assert net.model.lr == lr


@pytest.mark.parametrize("loss_strategy", ["standard", "robust_divergence"])
def test_vae_classification(loss_strategy: str) -> None:
    X = get_airfoil_dataset()

    model = TabularVAE(
        X,
        n_units_embedding=50,
        n_iter=100,
        encoder_max_clusters=5,
        loss_strategy=loss_strategy,
    )
    model.fit(X)

    generated = model.generate(10)

    assert (X.columns == generated.columns).all()
    assert generated.shape == (10, X.shape[1])


@pytest.mark.parametrize("loss_strategy", ["standard", "robust_divergence"])
@pytest.mark.skipif(sys.version_info < (3, 8), reason="test with python3.8 or higher")
def test_vae_classification_early_stopping(loss_strategy: str) -> None:
    X = get_airfoil_dataset()

    model = TabularVAE(
        X,
        n_units_embedding=50,
        n_iter=100,
        encoder_max_clusters=5,
        loss_strategy=loss_strategy,
        patience=1,
    )
    model.fit(X)

    generated = model.generate(10)

    assert (X.columns == generated.columns).all()
    assert generated.shape == (10, X.shape[1])
