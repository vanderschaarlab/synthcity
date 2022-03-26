# third party
import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_digits

# synthcity absolute
from synthcity.plugins.models.mlp import MLP


def test_network_config() -> None:
    net = MLP(
        task_type="regression",
        n_units_in=10,
        n_units_out=2,
        n_layers_hidden=2,
        n_units_hidden=20,
        batch_size=23,
        n_iter=34,
        lr=1e-2,
        dropout=0.5,
        batch_norm=True,
        nonlin="elu",
        patience=66,
        seed=77,
    )

    assert len(net.model) == 8
    assert net.batch_size == 23
    assert net.n_iter == 34
    assert net.lr == 1e-2
    assert net.patience == 66
    assert net.seed == 77


@pytest.mark.parametrize("task_type", ["regression", "classification"])
@pytest.mark.parametrize("nonlin", ["relu", "elu", "leaky_relu"])
@pytest.mark.parametrize("n_iter", [10, 50, 100])
@pytest.mark.parametrize("dropout", [0, 0.5, 0.2])
@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize("lr", [1e-3, 3e-4])
def test_basic_network(
    task_type: str,
    nonlin: str,
    n_iter: int,
    dropout: float,
    batch_norm: bool,
    lr: float,
) -> None:
    net = MLP(
        task_type=task_type,
        n_units_in=10,
        n_units_out=2,
        n_iter=n_iter,
        dropout=dropout,
        nonlin=nonlin,
        batch_norm=batch_norm,
        n_layers_hidden=2,
        lr=lr,
    )

    assert net.n_iter == n_iter
    assert net.task_type == task_type
    assert net.lr == lr


def test_mlp_classification() -> None:
    X, y = load_digits(return_X_y=True)
    model = MLP(
        task_type="classification", n_units_in=X.shape[1], n_units_out=len(np.unique(y))
    )

    model.fit(X, y)

    assert model.predict(X).shape == y.shape
    assert model.predict_proba(X).shape == (len(y), 10)


def test_mlp_regression() -> None:
    X, y = load_diabetes(return_X_y=True)
    model = MLP(task_type="regression", n_units_in=X.shape[1], n_units_out=1)

    model.fit(X, y)

    assert model.predict(X).shape == y.shape
    with pytest.raises(ValueError):
        model.predict_proba(X)
    print(model.score(X, y))
