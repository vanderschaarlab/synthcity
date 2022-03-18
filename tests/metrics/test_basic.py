# third party
import pytest
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics.basic import common_rows, integrity_score, nearest_synth_neighbor
from synthcity.plugins import Plugin, Plugins


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_integrity_score(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    score = integrity_score(
        X.drop(columns=["target"]),
        X["target"],
        X_gen.drop(columns=["target"]),
        X_gen["target"],
    )

    assert score == 0

    X_fail = X.head(100)
    X["target"] = "a"

    score = integrity_score(
        X.drop(columns=["target"]),
        X["target"],
        X_fail.drop(columns=["target"]),
        X_fail["target"],
    )

    assert score > 0


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_common_rows(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    score = common_rows(
        X.drop(columns=["target"]),
        X["target"],
        X_gen.drop(columns=["target"]),
        X_gen["target"],
    )

    assert score < 1


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_nearest_synth_neighbor(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    score = nearest_synth_neighbor(
        X.drop(columns=["target"]),
        X["target"],
        X_gen.drop(columns=["target"]),
        X_gen["target"],
    )

    assert score > 0
