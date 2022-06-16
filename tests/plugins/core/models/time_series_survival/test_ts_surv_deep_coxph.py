# third party
import numpy as np
import pytest

# synthcity absolute
from synthcity.plugins.core.models.time_series_survival.benchmarks import (
    evaluate_ts_survival_model,
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


def test_train_prediction() -> None:
    static, temporal, outcome = PBCDataloader(as_numpy=True).load()
    T, E, _, _ = outcome

    horizons = [0.85]
    time_horizons = np.quantile(
        [t_ for t_, e_ in zip(T, E) if e_ == 1], horizons
    ).tolist()

    model = DeepCoxPHTimeSeriesSurvival()
    score = evaluate_ts_survival_model(model, static, temporal, T, E, time_horizons)

    print(model.name(), score["str"])

    assert score["clf"]["c_index"][0] > 0.5
    assert score["clf"]["brier_score"][0] < 0.3


@pytest.mark.skip
def test_train_params() -> None:
    static, temporal, outcome = PBCDataloader(as_numpy=True).load()
    T, E, _, _ = outcome

    horizons = [0.85]
    time_horizons = np.quantile(
        [t_ for t_, e_ in zip(T, E) if e_ == 1], horizons
    ).tolist()

    params_list = []
    for t in range(100):
        params_list.append(DeepCoxPHTimeSeriesSurvival.sample_hyperparameters())

    best_c = 0
    for param in params_list:
        model = DeepCoxPHTimeSeriesSurvival(**param)
        score = evaluate_ts_survival_model(model, static, temporal, T, E, time_horizons)

        if score["clf"]["c_index"][0] > best_c:
            best_c = score["clf"]["c_index"][0]
            print(param, score["str"])
