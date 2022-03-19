# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics.performance import performance_train_synth_classifier
from synthcity.plugins import Plugin, Plugins


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_performance_train_synth_classifier(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    good_score = performance_train_synth_classifier(
        X.drop(columns=["target"]),
        X["target"],
        X_gen.drop(columns=["target"]),
        X_gen["target"],
    )

    assert np.abs(good_score) < 1

    sz = 100
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    score = performance_train_synth_classifier(
        X.drop(columns=["target"]),
        X["target"],
        X_rnd.drop(columns=["target"]),
        X_rnd["target"],
    )

    assert np.abs(good_score) < 1
    assert score > good_score
