# third party
import numpy as np

# synthcity absolute
from synthcity.plugins.core.models.time_series_survival.metrics import (
    evaluate_brier_score_ts,
    evaluate_c_index_ts,
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

    assert len(params.keys()) == 4


def test_train_prediction() -> None:
    static, temporal, outcome = PBCDataloader().load()
    T, E = outcome

    tr_size = int(len(temporal) * 0.80)

    static_train, static_test = static[:tr_size], static[-tr_size:]
    temporal_train, temporal_test = temporal[:tr_size], temporal[-tr_size:]
    T_train, T_test = T[:tr_size], T[-tr_size:]
    E_train, E_test = E[:tr_size], E[-tr_size:]

    horizons = [0.25, 0.5, 0.75]
    time_horizons = np.quantile(
        [t_[-1] for t_, e_ in zip(T, E) if e_[-1] == 1], horizons
    ).tolist()

    model = DynamicDeephitTimeSeriesSurvival()

    model.fit(static_train, temporal_train, T_train, E_train)

    prediction = model.predict(static_test, temporal_test, time_horizons)

    assert prediction.shape == (len(temporal_test), len(time_horizons))
    assert (prediction.columns == time_horizons).any()

    horizon = time_horizons[0]
    cindex = evaluate_c_index_ts(
        T_train,
        E_train,
        prediction[horizon].values,
        T_test,
        E_test,
        horizon,
        use_last=True,
    )
    bs = evaluate_brier_score_ts(
        T_train,
        E_train,
        prediction[horizon].values,
        T_test,
        E_test,
        horizon,
        use_last=True,
    )
    print(model.name(), cindex, bs)

    assert cindex > 0.7
    assert bs < 0.2
