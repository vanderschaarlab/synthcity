# third party
from lifelines.datasets import load_rossi

# synthcity absolute
from synthcity.plugins.core.models.time_to_event.tte_coxph import CoxPHTimeToEvent


def test_sanity() -> None:
    model = CoxPHTimeToEvent()

    assert model.name() == "cox_ph"


def test_hyperparams() -> None:
    model = CoxPHTimeToEvent()

    params = model.sample_hyperparameters()

    assert len(params.keys()) == 2


def test_train_prediction() -> None:
    df = load_rossi()

    X = df.drop(["week", "arrest"], axis=1)
    Y = df["arrest"]
    T = df["week"]

    model = CoxPHTimeToEvent()

    model.fit(X, T, Y)

    prediction = model.predict(X)

    assert prediction.shape == T.shape
