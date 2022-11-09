# third party
import numpy as np
import pytest

# synthcity absolute
from synthcity.plugins.core.models.time_series_survival.benchmarks import (
    evaluate_ts_survival_model,
    search_hyperparams,
)
from synthcity.plugins.core.models.time_series_survival.ts_surv_deep_coxph import (
    DeepCoxPHTimeSeriesSurvival,
)
from synthcity.utils.datasets.time_series.pbc import PBCDataloader


def test_sanity() -> None:
    model = DeepCoxPHTimeSeriesSurvival()

    assert model.name() == "deep_recurrent_coxph"


def test_hyperparams() -> None:
    model = DeepCoxPHTimeSeriesSurvival()

    params = model.sample_hyperparameters()

    assert len(params.keys()) == 6


@pytest.mark.parametrize("rnn_type", ["GRU", "LSTM", "Transformer"])
@pytest.mark.slow
def test_train_prediction(rnn_type: str) -> None:
    static, temporal, temporal_horizons, outcome = PBCDataloader(as_numpy=True).load()
    T, E = outcome

    horizons = [0.25, 0.5, 0.75]
    time_horizons = np.quantile(T, horizons).tolist()

    model = DeepCoxPHTimeSeriesSurvival(rnn_type=rnn_type)
    score = evaluate_ts_survival_model(
        model, static, temporal, temporal_horizons, T, E, time_horizons
    )

    print("Perf", model.name(), score["str"])
    assert score["clf"]["c_index"][0] > 0


@pytest.mark.slow
def test_hyperparams_search() -> None:
    static, temporal, temporal_horizons, outcome = PBCDataloader(as_numpy=True).load()
    T, E = outcome

    horizons = [0.25, 0.5, 0.75]
    time_horizons = np.quantile(
        [t_ for t_, e_ in zip(T, E) if e_ == 1], horizons
    ).tolist()

    args = search_hyperparams(
        DeepCoxPHTimeSeriesSurvival,
        static,
        temporal,
        temporal_horizons,
        T,
        E,
        time_horizons,
    )

    model = DeepCoxPHTimeSeriesSurvival(**args)
    score = evaluate_ts_survival_model(
        model, static, temporal, temporal_horizons, T, E, time_horizons
    )

    print("Perf", model.name(), args, score["str"])
    assert score["clf"]["c_index"][0] > 0
