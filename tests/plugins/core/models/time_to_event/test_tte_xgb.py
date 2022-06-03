# third party
from lifelines.datasets import load_rossi

# synthcity absolute
from synthcity.plugins.core.models.time_to_event.tte_xgb import XGBTimeToEvent


def test_sanity() -> None:
    model = XGBTimeToEvent()

    assert model.name() == "survival_xgboost"


def test_hyperparams() -> None:
    model = XGBTimeToEvent()

    params = model.sample_hyperparameters()

    assert len(params.keys()) == 5


def test_train_prediction() -> None:
    df = load_rossi()

    X = df.drop(["week", "arrest"], axis=1)
    Y = df["arrest"]
    T = df["week"]

    model = XGBTimeToEvent(n_estimators=10)

    model.fit(X, T, Y)

    prediction = model.predict(X)

    assert prediction.shape == T.shape
