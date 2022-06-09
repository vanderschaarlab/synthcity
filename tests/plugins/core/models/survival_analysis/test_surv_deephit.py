# third party
import numpy as np
from lifelines.datasets import load_rossi

# synthcity absolute
from synthcity.plugins.core.models.survival_analysis.surv_deephit import (
    DeephitSurvivalAnalysis,
)


def test_sanity() -> None:
    model = DeephitSurvivalAnalysis()

    assert model.name() == "deephit"


def test_hyperparams() -> None:
    model = DeephitSurvivalAnalysis()

    params = model.sample_hyperparameters()

    assert len(params.keys()) == 7


def test_train_prediction() -> None:
    df = load_rossi()

    X = df.drop(["week", "arrest"], axis=1)
    Y = df["arrest"]
    T = df["week"]
    time_horizons = np.linspace(T.min(), T.max(), num=4)[1:-1].tolist()

    model = DeephitSurvivalAnalysis()

    model.fit(X, T, Y)

    prediction = model.predict(X, time_horizons)

    assert prediction.shape == (len(X), len(time_horizons))
    assert (prediction.columns == time_horizons).any()
