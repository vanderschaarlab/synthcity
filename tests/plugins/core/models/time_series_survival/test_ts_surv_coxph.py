# third party
import numpy as np
import pytest

# synthcity absolute
from synthcity.plugins.core.models.time_series_survival.benchmarks import (
    evaluate_ts_survival_model,
    search_hyperparams,
)
from synthcity.plugins.core.models.time_series_survival.ts_surv_coxph import (
    CoxTimeSeriesSurvival,
)
from synthcity.utils.datasets.time_series.pbc import PBCDataloader


def test_sanity() -> None:
    model = CoxTimeSeriesSurvival()

    assert model.name() == "ts_coxph"


def test_hyperparams() -> None:
    model = CoxTimeSeriesSurvival()

    params = model.sample_hyperparameters()

    assert len(params.keys()) == 11


def test_train_prediction() -> None:
    static, temporal, outcome = PBCDataloader(as_numpy=True).load()
    T, E, _, _ = outcome

    horizons = [0.25, 0.5, 0.75]
    time_horizons = np.quantile(T, horizons).tolist()

    model = CoxTimeSeriesSurvival()
    score = evaluate_ts_survival_model(model, static, temporal, T, E, time_horizons)

    print("Perf", model.name(), score["str"])
    assert score["clf"]["c_index"][0] > 0.5


@pytest.mark.slow
def test_hyperparam_search() -> None:
    static, temporal, outcome = PBCDataloader(as_numpy=True).load()
    T, E, _, _ = outcome

    horizons = [0.25, 0.5, 0.75]
    time_horizons = np.quantile(
        [t_ for t_, e_ in zip(T, E) if e_ == 1], horizons
    ).tolist()

    args = search_hyperparams(
        CoxTimeSeriesSurvival, static, temporal, T, E, time_horizons
    )

    model = CoxTimeSeriesSurvival(**args)
    score = evaluate_ts_survival_model(model, static, temporal, T, E, time_horizons)

    print("Perf", model.name(), args, score["str"])
    assert score["clf"]["c_index"][0] > 0.5
