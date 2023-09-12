# third party
from lifelines.datasets import load_rossi

# synthcity absolute
from synthcity.plugins.core.models.time_to_event.tte_deephit import DeephitTimeToEvent


def test_sanity() -> None:
    model = DeephitTimeToEvent()

    assert model.name() == "deephit"


def test_hyperparams() -> None:
    model = DeephitTimeToEvent()

    params = model.sample_hyperparameters()

    assert len(params.keys()) == 7


def test_train_prediction() -> None:
    df = load_rossi()

    X = df.drop(["week", "arrest"], axis=1)
    Y = df["arrest"]
    T = df["week"]

    model = DeephitTimeToEvent()

    model.fit(X, T, Y)

    prediction = model.predict(X)

    assert prediction.shape == T.shape
