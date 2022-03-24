# stdlib
from typing import Callable

# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics.detection import (
    evaluate_gmm_detection_synthetic,
    evaluate_mlp_detection_synthetic,
    evaluate_xgb_detection_synthetic,
)
from synthcity.plugins import Plugin, Plugins


@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
@pytest.mark.parametrize(
    "method",
    [
        evaluate_xgb_detection_synthetic,
        evaluate_mlp_detection_synthetic,
        evaluate_gmm_detection_synthetic,
    ],
)
def test_detect_synth(test_plugin: Plugin, method: Callable) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(100)

    good_score = method(
        X,
        X_gen,
    )

    assert good_score > 0
    assert good_score <= 1

    sz = 100
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    score = method(
        X,
        X_rnd,
    )

    assert score > 0
    assert score <= 1
    assert good_score < score
