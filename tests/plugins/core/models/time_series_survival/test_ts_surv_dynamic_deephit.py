# third party
import numpy as np

# synthcity absolute
from synthcity.plugins.core.models.time_series_survival.benchmarks import (
    evaluate_ts_survival_model,
    search_hyperparams,
)
from synthcity.plugins.core.models.time_series_survival.ts_surv_dynamic_deephit import (
    DynamicDeephitTimeSeriesSurvival,
)
from synthcity.utils.datasets.time_series.pbc import PBCDataloader


def test_sanity() -> None:
    model = DynamicDeephitTimeSeriesSurvival()

    assert model.name() == "dynamic_deephit"


def test_hyperparams() -> None:
    model = DynamicDeephitTimeSeriesSurvival()

    params = model.sample_hyperparameters()

    assert len(params.keys()) == 9


def test_train_prediction() -> None:
    static, temporal, outcome = PBCDataloader(as_numpy=True).load()
    T, E, _, _ = outcome

    horizons = [0.25, 0.5, 0.75]
    time_horizons = np.quantile(
        [t_ for t_, e_ in zip(T, E) if e_ == 1], horizons
    ).tolist()

    args = search_hyperparams(
        DynamicDeephitTimeSeriesSurvival, static, temporal, T, E, time_horizons
    )

    model = DynamicDeephitTimeSeriesSurvival(**args)
    score = evaluate_ts_survival_model(model, static, temporal, T, E, time_horizons)

    print("Perf", model.name(), score["str"])
    assert score["clf"]["c_index"][0] > 0.5
