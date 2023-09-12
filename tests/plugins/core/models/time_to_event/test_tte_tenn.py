# third party
from lifelines.datasets import load_rossi

# synthcity absolute
from synthcity.plugins.core.models.time_to_event.tte_tenn import TENNTimeToEvent


def test_sanity() -> None:
    model = TENNTimeToEvent()

    assert model.name() == "tenn"


def test_hyperparams() -> None:
    model = TENNTimeToEvent()

    params = model.sample_hyperparameters()

    assert len(params.keys()) == 10


def test_train_prediction() -> None:
    df = load_rossi()

    X = df.drop(["week", "arrest"], axis=1)
    Y = df["arrest"]
    T = df["week"]

    model = TENNTimeToEvent(n_iter=100)

    model.fit(X, T, Y)

    prediction = model.predict(X)

    assert prediction.shape == T.shape
