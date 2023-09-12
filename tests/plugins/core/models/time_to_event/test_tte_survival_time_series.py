# third party
import numpy as np
import pytest
from sklearn.metrics import mean_squared_error

# synthcity absolute
from synthcity.plugins.core.models.time_to_event.tte_survival_time_series import (
    TSSurvivalFunctionTimeToEvent,
)
from synthcity.utils.datasets.time_series.pbc import PBCDataloader


def test_sanity() -> None:
    model = TSSurvivalFunctionTimeToEvent()

    assert model.name() == "ts_survival_function_regression"


def test_hyperparams() -> None:
    model = TSSurvivalFunctionTimeToEvent()

    params = model.sample_hyperparameters()

    assert len(params.keys()) == 1


@pytest.mark.parametrize("survival_base_learner", ["RNN", "Transformer", "LSTM", "GRU"])
@pytest.mark.skip
def test_train_prediction(survival_base_learner: str) -> None:
    static, temporal, observation_times, outcome = PBCDataloader(as_numpy=True).load()
    T, E = outcome

    observation_times = np.asarray(observation_times)

    model = TSSurvivalFunctionTimeToEvent(
        survival_base_learner=survival_base_learner, n_iter=10
    )

    model.fit(static, temporal, observation_times, T, E)

    prediction = model.predict(static, temporal, observation_times)

    assert prediction.shape == T.shape

    prediction_any = model.predict_any(static, temporal, observation_times, E)

    assert prediction_any.shape == T.shape
    print("error", survival_base_learner, mean_squared_error(T, prediction_any))
