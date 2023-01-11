# third party
import numpy as np
import pytest

# synthcity absolute
from synthcity.plugins.core.models.time_series_survival.benchmarks import (
    evaluate_ts_survival_model,
    search_hyperparams,
)
from synthcity.plugins.core.models.time_series_survival.ts_surv_xgb import (
    XGBTimeSeriesSurvival,
)
from synthcity.utils.datasets.time_series.pbc import PBCDataloader


def test_sanity() -> None:
    model = XGBTimeSeriesSurvival()

    assert model.name() == "ts_xgb"


def test_hyperparams() -> None:
    model = XGBTimeSeriesSurvival()

    params = model.sample_hyperparameters()

    assert len(params.keys()) == 23


@pytest.mark.parametrize("emb_rnn_type", ["Transformer"])
def test_train_prediction(emb_rnn_type: str) -> None:
    static, temporal, observation_times, outcome = PBCDataloader(as_numpy=True).load()
    T, E = outcome

    horizons = [0.25, 0.5, 0.75]
    time_horizons = np.quantile(T, horizons).tolist()

    model = XGBTimeSeriesSurvival(emb_rnn_type=emb_rnn_type)
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
        XGBTimeSeriesSurvival, static, temporal, observation_times, T, E, time_horizons
    )

    model = XGBTimeSeriesSurvival(**args)
    score = evaluate_ts_survival_model(
        model, static, temporal, observation_times, T, E, time_horizons
    )

    print("Perf", model.name(), args, score["str"])
    assert score["clf"]["c_index"][0] > 0.5
