# third party
from lifelines.datasets import load_rossi

# synthcity absolute
from synthcity.plugins.core.models.time_to_event.tte_survival_function_regression import (
    SurvivalFunctionTimeToEvent,
)


def test_sanity() -> None:
    model = SurvivalFunctionTimeToEvent()

    assert model.name() == "survival_function_regression"


def test_hyperparams() -> None:
    model = SurvivalFunctionTimeToEvent()

    params = model.sample_hyperparameters()

    assert len(params.keys()) == 1


def test_train_prediction() -> None:
    df = load_rossi()

    X = df.drop(["week", "arrest"], axis=1)
    Y = df["arrest"]
    T = df["week"]

    model = SurvivalFunctionTimeToEvent()

    model.fit(X, T, Y)

    prediction = model.predict(X)

    assert prediction.shape == T.shape
