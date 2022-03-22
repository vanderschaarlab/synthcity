# third party
import pandas as pd
import pytest
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics import evaluate
from synthcity.plugins import Plugins


@pytest.mark.parametrize("test_plugin", Plugins().list())
def test_basic(test_plugin: str) -> None:
    model = Plugins().get(test_plugin)

    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    model.fit(X)
    X_gen = model.generate(100)

    out = evaluate(
        X.drop(columns=["target"]),
        X["target"],
        X_gen.drop(columns=["target"]),
        X_gen["target"],
    )

    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == set(
        ["mean", "min", "max", "median", "iqr", "stddev", "rounds", "durations"]
    )
