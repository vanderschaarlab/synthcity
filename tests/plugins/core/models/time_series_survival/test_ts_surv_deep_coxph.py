# third party
import numpy as np

# synthcity absolute
from synthcity.plugins.core.models.survival_analysis.metrics import (
    evaluate_brier_score,
    evaluate_c_index,
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

    assert len(params.keys()) == 4


def test_train_prediction() -> None:
    static, temporal, outcome = PBCDataloader(as_numpy=True).load()
    T, E, T_ext, E_ext = outcome

    tr_size = int(len(temporal) * 0.80)

    static_train, static_test = static[:tr_size], static[-tr_size:]
    temporal_train, temporal_test = temporal[:tr_size], temporal[-tr_size:]
    T_train, T_test = T[:tr_size], T[-tr_size:]
    T_ext_train, _ = T_ext[:tr_size], T_ext[-tr_size:]
    E_train, E_test = E[:tr_size], E[-tr_size:]
    E_ext_train, _ = E_ext[:tr_size], E_ext[-tr_size:]

    horizons = [0.25, 0.5, 0.75]
    time_horizons = np.quantile(
        [t_ for t_, e_ in zip(T, E) if e_ == 1], horizons
    ).tolist()

    model = DeepCoxPHTimeSeriesSurvival()

    model.fit(static_train, temporal_train, T_ext_train, E_ext_train)

    prediction = model.predict(static_test, temporal_test, time_horizons)

    assert prediction.shape == (len(temporal_test), len(time_horizons))
    assert (prediction.columns == time_horizons).any()

    for horizon in time_horizons:
        cindex = evaluate_c_index(
            T_train, E_train, prediction[horizon].values, T_test, E_test, horizon
        )
        bs = evaluate_brier_score(
            T_train, E_train, prediction[horizon].values, T_test, E_test, horizon
        )
        print(
            f"Model = {model.name()}. Horizon = {horizon}. cindex = {cindex}. brier_score = {bs}"
        )

        assert cindex > 0.4
        assert bs < 0.3
