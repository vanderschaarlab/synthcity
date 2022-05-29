# third party
import numpy as np
from lifelines.datasets import load_rossi

# synthcity absolute
from synthcity.plugins.core.models.survival_analysis.metrics import (
    km_survival_function,
    nonparametric_distance,
)


def test_km_surv_function() -> None:
    df = load_rossi()

    E = df["arrest"]
    T = df["week"]

    kmf, surv_fn, hazards, constant_hazard = km_survival_function(T, E)

    assert kmf.predict(T).shape == E.shape
    assert surv_fn.shape == hazards.shape
    assert constant_hazard < 1


def test_nonparametric_distance() -> None:
    df = load_rossi()

    E = df["arrest"]
    T = df["week"]

    ind = np.random.choice(T.unique(), size=(5,), replace=False)

    auc_opt, auc_abs_opt, sightedness = nonparametric_distance(
        (T[~T.isin(ind)], E[~T.isin(ind)]), (T, E)
    )

    assert -1 < auc_opt < 1
    assert 0 < auc_abs_opt < 1

    assert sightedness < 1
