# third party
from lifelines.datasets import load_rossi

# synthcity absolute
from synthcity.plugins.core.models.time_to_event.tte_date import DATETimeToEvent


def test_sanity() -> None:
    model = DATETimeToEvent()

    assert model.name() == "date"


def test_hyperparams() -> None:
    model = DATETimeToEvent()

    params = model.sample_hyperparameters()

    assert len(params.keys()) == 12


def test_train_prediction() -> None:
    df = load_rossi()

    X = df.drop(["week", "arrest"], axis=1)
    Y = df["arrest"]
    T = df["week"]

    model = DATETimeToEvent(generator_n_iter=100)

    model.fit(X, T, Y)

    prediction = model.predict(X)

    assert prediction.shape == T.shape
