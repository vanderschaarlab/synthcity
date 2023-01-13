# third party
import numpy as np
import pytest

# synthcity absolute
from synthcity.plugins.core.models.time_series_survival.benchmarks import (
    evaluate_ts_survival_model,
    search_hyperparams,
)
from synthcity.plugins.core.models.time_series_survival.ts_surv_dynamic_deephit import (
    DynamicDeephitTimeSeriesSurvival,
    rnn_modes,
)
from synthcity.utils.datasets.time_series.pbc import PBCDataloader


def test_sanity() -> None:
    model = DynamicDeephitTimeSeriesSurvival()

    assert model.name() == "dynamic_deephit"


def test_hyperparams() -> None:
    model = DynamicDeephitTimeSeriesSurvival()

    params = model.sample_hyperparameters()

    assert len(params.keys()) == 10


@pytest.mark.parametrize("rnn_type", rnn_modes)
def test_train(rnn_type: str) -> None:
    static, temporal, observation_times, outcome = PBCDataloader(as_numpy=True).load()
    T, E = outcome

    horizons = [0.25, 0.5, 0.75]
    time_horizons = np.quantile(T, horizons).tolist()

    model = DynamicDeephitTimeSeriesSurvival(rnn_type=rnn_type)
    model.fit(static, temporal, observation_times, T, E)
    out = model.predict(
        static, temporal, observation_times, time_horizons=time_horizons
    )

    assert out.shape == (len(temporal), len(time_horizons))


@pytest.mark.parametrize("rnn_type", ["LSTM", "Transformer"])
@pytest.mark.parametrize("output_type", ["MLP"])
def test_train_prediction_dyn_deephit(rnn_type: str, output_type: str) -> None:
    static, temporal, observation_times, outcome = PBCDataloader(as_numpy=True).load()
    T, E = outcome

    horizons = [0.25, 0.5, 0.75]
    time_horizons = np.quantile(T, horizons).tolist()

    model = DynamicDeephitTimeSeriesSurvival(rnn_type=rnn_type, output_type=output_type)
    score = evaluate_ts_survival_model(
        model, static, temporal, observation_times, T, E, time_horizons
    )

    print("Perf", model.name(), score["str"])
    assert score["clf"]["c_index"][0] > 0.5


@pytest.mark.slow
def test_hyperparam_search() -> None:
    static, temporal, observation_times, outcome = PBCDataloader(as_numpy=True).load()
    T, E = outcome

    horizons = [0.25, 0.5, 0.75]
    time_horizons = np.quantile(
        [t_ for t_, e_ in zip(T, E) if e_ == 1], horizons
    ).tolist()

    args = search_hyperparams(
        DynamicDeephitTimeSeriesSurvival,
        static,
        temporal,
        observation_times,
        T,
        E,
        time_horizons,
    )

    model = DynamicDeephitTimeSeriesSurvival(**args)
    score = evaluate_ts_survival_model(
        model, static, temporal, observation_times, T, E, time_horizons
    )

    print("Perf", model.name(), score["str"])
    assert score["clf"]["c_index"][0] > 0.5
