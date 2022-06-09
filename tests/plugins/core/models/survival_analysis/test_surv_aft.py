# third party
import numpy as np
from lifelines.datasets import load_rossi

# synthcity absolute
from synthcity.plugins.core.models.survival_analysis.surv_aft import (
    WeibullAFTSurvivalAnalysis,
)


def test_sanity() -> None:
    model = WeibullAFTSurvivalAnalysis()

    assert model.name() == "weibull_aft"


def test_hyperparams() -> None:
    model = WeibullAFTSurvivalAnalysis()

    params = model.sample_hyperparameters()

    assert len(params.keys()) == 2


def test_train_prediction() -> None:
    df = load_rossi()

    X = df.drop(["week", "arrest"], axis=1)
    Y = df["arrest"]
    T = df["week"]
    time_horizons = np.linspace(T.min(), T.max(), num=4)[1:-1].tolist()

    model = WeibullAFTSurvivalAnalysis()

    model.fit(X, T, Y)

    prediction = model.predict(X, time_horizons)

    assert prediction.shape == (len(X), len(time_horizons))
    assert (prediction.columns == time_horizons).any()
