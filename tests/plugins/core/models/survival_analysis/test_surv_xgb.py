# third party
import numpy as np
from lifelines.datasets import load_rossi

# synthcity absolute
from synthcity.plugins.core.models.survival_analysis.surv_xgb import XGBSurvivalAnalysis


def test_sanity() -> None:
    model = XGBSurvivalAnalysis()

    assert model.name() == "survival_xgboost"


def test_hyperparams() -> None:
    model = XGBSurvivalAnalysis()

    params = model.sample_hyperparameters()

    assert len(params.keys()) == 13


def test_train_prediction() -> None:
    df = load_rossi()

    X = df.drop(["week", "arrest"], axis=1)
    Y = df["arrest"]
    T = df["week"]
    time_horizons = np.linspace(T.min(), T.max(), num=4)[1:-1].tolist()

    model = XGBSurvivalAnalysis(n_estimators=10)

    model.fit(X, T, Y)

    prediction = model.predict(X, time_horizons)

    assert prediction.shape == (len(X), len(time_horizons))
    assert (prediction.columns == time_horizons).any()
