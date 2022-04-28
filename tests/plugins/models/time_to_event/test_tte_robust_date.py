# third party
from lifelines.datasets import load_rossi

# synthcity absolute
from synthcity.plugins.models.time_to_event.tte_robust_date import RobustDATETimeToEvent


def test_sanity() -> None:
    model = RobustDATETimeToEvent()

    assert model.name() == "robust_date"


def test_hyperparams() -> None:
    model = RobustDATETimeToEvent()

    params = model.sample_hyperparameters()

    assert len(params.keys()) == 15


def test_train_prediction() -> None:
    df = load_rossi()

    X = df.drop(["week", "arrest"], axis=1)
    Y = df["arrest"]
    T = df["week"]

    model = RobustDATETimeToEvent(generator_n_iter=100)

    model.fit(X, T, Y)

    prediction = model.predict(X)

    assert prediction.shape == T.shape
