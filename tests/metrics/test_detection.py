# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics.detection import evaluate_detection_synthetic
from synthcity.plugins import Plugin, Plugins


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_detect_synth(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    good_score = evaluate_detection_synthetic(
        X.drop(columns=["target"]),
        X["target"],
        X_gen.drop(columns=["target"]),
        X_gen["target"],
    )

    assert good_score > 0
    assert good_score <= 1

    sz = 100
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    score = evaluate_detection_synthetic(
        X.drop(columns=["target"]),
        X["target"],
        X_rnd.drop(columns=["target"]),
        X_rnd["target"],
    )

    assert score > 0
    assert score <= 1
    assert good_score < score
