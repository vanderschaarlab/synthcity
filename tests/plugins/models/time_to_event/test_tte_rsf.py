# third party
from lifelines.datasets import load_rossi

# synthcity absolute
from synthcity.plugins.models.time_to_event.tte_rsf import (
    RandomSurvivalForestTimeToEvent,
)


def test_sanity() -> None:
    model = RandomSurvivalForestTimeToEvent()

    assert model.name() == "random_survival_forest"


def test_hyperparams() -> None:
    model = RandomSurvivalForestTimeToEvent()

    params = model.sample_hyperparameters()

    assert len(params.keys()) == 0


def test_train_prediction() -> None:
    df = load_rossi()

    X = df.drop(["week", "arrest"], axis=1)
    Y = df["arrest"]
    T = df["week"]

    model = RandomSurvivalForestTimeToEvent()

    model.fit(X, T, Y)

    prediction = model.predict(X)

    assert prediction.shape == T.shape
