# third party
import numpy as np
import pytest

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


@pytest.mark.parametrize("base_learner", ["RNN", "Transformer", "LSTM", "GRU"])
def test_train_prediction(base_learner: str) -> None:
    static, temporal, temporal_horizons, outcome = PBCDataloader(as_numpy=True).load()
    T, E = outcome

    temporal_horizons = np.asarray(temporal_horizons)

    horizons = [0.25, 0.5, 0.75]
    time_horizons = np.quantile(T, horizons).tolist()

    model = TSSurvivalFunctionTimeToEvent(base_learner=base_learner)

    model.fit(static, temporal, temporal_horizons, T, E)

    prediction = model.predict(
        static, temporal, temporal_horizons, time_horizons=time_horizons
    )

    assert prediction.shape == T.shape

    prediction_any = model.predict_any(static, temporal, temporal_horizons, E)

    print(prediction_any, T)
    assert prediction_any.shape == T.shape
