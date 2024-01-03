# third party
from lifelines.datasets import load_rossi

# synthcity absolute
from synthcity.plugins.core.models.time_to_event.tte_aft import WeibullAFTTimeToEvent


def test_sanity() -> None:
    model = WeibullAFTTimeToEvent()

    assert model.name() == "weibull_aft"


def test_hyperparams() -> None:
    model = WeibullAFTTimeToEvent()

    params = model.sample_hyperparameters()

    assert len(params.keys()) == 2


def test_train_prediction() -> None:
    df = load_rossi()

    X = df.drop(["week", "arrest"], axis=1)
    Y = df["arrest"]
    T = df["week"]

    model = WeibullAFTTimeToEvent()

    model.fit(X, T, Y)

    prediction = model.predict(X)

    assert prediction.shape == T.shape
