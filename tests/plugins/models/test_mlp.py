# third party
import jax
import pytest

# synthcity absolute
from synthcity.plugins.models.mlp import BasicNetwork, NetworkConfig, create_train_state


def test_network_config() -> None:
    config = NetworkConfig(
        task_type="regression",
        model_type=str,
        input_shape=12,
        output_shape=11,
        hidden_layers=[1, 2, 3, 4],
        batch_size=23,
        epochs=34,
        learning_rate=1e-2,
        dropout=0.5,
        batchnorm=True,
        nonlin="elu",
        patience=66,
        seed=77,
        optimizer="sgd",
    )

    assert config.model_type == str
    assert config.input_shape == 12
    assert config.output_shape == 11
    assert list(config.hidden_layers) == [1, 2, 3, 4]
    assert config.batch_size == 23
    assert config.epochs == 34
    assert config.learning_rate == 1e-2
    assert config.dropout == 0.5
    assert config.batchnorm is True
    assert config.nonlin == "elu"
    assert config.patience == 66
    assert config.seed == 77
    assert config.optimizer == "sgd"


@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
@pytest.mark.parametrize("task_type", ["regression", "classification"])
@pytest.mark.parametrize("nonlin", ["relu", "elu", "leaky_relu"])
@pytest.mark.parametrize("epochs", [10, 50, 100])
@pytest.mark.parametrize("dropout", [0, 0.5, 0.2])
@pytest.mark.parametrize("batchnorm", [True, False])
def test_basic_network(
    optimizer: str,
    task_type: str,
    nonlin: str,
    epochs: int,
    dropout: float,
    batchnorm: bool,
) -> None:
    config = NetworkConfig(
        task_type=task_type,
        model_type=BasicNetwork,
        epochs=epochs,
        dropout=dropout,
        nonlin=nonlin,
        input_shape=5,
        batchnorm=batchnorm,
        output_shape=2,
        hidden_layers=[1, 2],
        optimizer=optimizer,
    )

    rng = jax.random.PRNGKey(config.seed)
    state = create_train_state(config, rng)

    assert state is not None

    assert config.epochs == epochs
    assert config.dropout == dropout
    assert config.batchnorm == batchnorm
    assert config.nonlin == nonlin
    assert config.optimizer == optimizer
    assert config.task_type == task_type
