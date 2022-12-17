# stdlib
from typing import Tuple

# third party
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy.integrate import trapz
from xgbse.non_parametric import _get_conditional_probs_from_survival

# synthcity absolute
from synthcity.plugins.core.models.survival_analysis.third_party.metrics import (
    brier_score,
    concordance_index_ipcw,
)


def evaluate_c_index(
    T_train: np.ndarray,
    Y_train: np.ndarray,
    Prediction: np.ndarray,
    T_test: np.ndarray,
    Y_test: np.ndarray,
    Time: float,
) -> float:
    """Helper for evaluating the C-INDEX metric."""
    T_train = pd.Series(T_train)
    Y_train = pd.Series(Y_train)
    T_test = pd.Series(T_test)
    Y_test = pd.Series(Y_test)
    Prediction = np.asarray(Prediction).squeeze()

    Y_train_structured = [
        (Y_train.iloc[i], T_train.iloc[i]) for i in range(len(Y_train))
    ]
    Y_train_structured = np.array(
        Y_train_structured, dtype=[("status", "bool"), ("time", "<f8")]
    )

    Y_test_structured = [(Y_test.iloc[i], T_test.iloc[i]) for i in range(len(Y_test))]
    Y_test_structured = np.array(
        Y_test_structured, dtype=[("status", "bool"), ("time", "<f8")]
    )

    # concordance_index_ipcw expects risk scores
    return concordance_index_ipcw(
        Y_train_structured, Y_test_structured, Prediction, tau=Time
    )


def evaluate_brier_score(
    T_train: np.ndarray,
    Y_train: np.ndarray,
    Prediction: np.ndarray,
    T_test: np.ndarray,
    Y_test: np.ndarray,
    Time: float,
) -> float:
    """Helper for evaluating the Brier score."""
    T_train = pd.Series(T_train)
    Y_train = pd.Series(Y_train)
    T_test = pd.Series(T_test)
    Y_test = pd.Series(Y_test)

    Y_train_structured = [
        (Y_train.iloc[i], T_train.iloc[i]) for i in range(len(Y_train))
    ]
    Y_train_structured = np.array(
        Y_train_structured, dtype=[("status", "bool"), ("time", "<f8")]
    )

    Y_test_structured = [(Y_test.iloc[i], T_test.iloc[i]) for i in range(len(Y_test))]
    Y_test_structured = np.array(
        Y_test_structured, dtype=[("status", "bool"), ("time", "<f8")]
    )

    # brier_score expects survival scores
    return brier_score(
        Y_train_structured, Y_test_structured, 1 - Prediction, times=Time
    )[0]


def km_survival_function(
    T: np.ndarray, E: np.ndarray
) -> Tuple[KaplanMeierFitter, np.ndarray, np.ndarray, np.ndarray]:
    kmf = KaplanMeierFitter().fit(T, E)
    surv_fn = kmf.survival_function_.T.reset_index(drop=True)
    if len(surv_fn.columns) < 2:
        raise RuntimeError("invalid survival functin for extrapolation")

    hazards = _get_conditional_probs_from_survival(surv_fn)
    constant_hazard = hazards.values[:, -1:].mean(axis=1)[0]

    return kmf, surv_fn, hazards, constant_hazard


def nonparametric_distance(
    real: Tuple[np.ndarray, np.ndarray],
    syn: Tuple[np.ndarray, np.ndarray],
    n_points: int = 1000,
) -> Tuple:
    real_T, real_E = real
    syn_T, syn_E = syn

    Tmax = max(real_T.max(), syn_T.max())
    Tmin = min(real_T.min(), syn_T.min())
    Tmin = max(0, Tmin)

    time_points = np.linspace(Tmin, Tmax, n_points)

    opt: list = []
    abs_opt: list = []

    real_kmf, real_surv, real_hazards, real_constant_hazard = km_survival_function(
        real_T, real_E
    )
    if len(syn) == 0 or len(real) == 0:
        raise ValueError("Empty evaluation sets")

    syn_kmf, syn_surv, syn_hazards, syn_constant_hazard = km_survival_function(
        syn_T, syn_E
    )

    abs_opt = []
    opt = []
    for t in time_points:
        syn_local_pred = syn_kmf.predict(t)
        real_local_pred = real_kmf.predict(t)

        if np.isnan(syn_local_pred):
            raise RuntimeError("syn_local_pred contains NaNs")
        if np.isnan(real_local_pred):
            raise RuntimeError("real_local_pred contains NaNs")

        abs_opt.append(abs(syn_local_pred - real_local_pred))
        opt.append(syn_local_pred - real_local_pred)

    auc_abs_opt = trapz(abs_opt, time_points) / Tmax
    auc_opt = trapz(opt, time_points) / Tmax
    sightedness = (real_T.max() - syn_T.max()) / Tmax

    return auc_opt, auc_abs_opt, sightedness


def generate_score(metric: np.ndarray) -> Tuple[float, float]:
    percentile_val = 1.96
    return (np.mean(metric), percentile_val * np.std(metric) / np.sqrt(len(metric)))


def print_score(score: Tuple[float, float]) -> str:
    return str(round(score[0], 4)) + " +/- " + str(round(score[1], 4))
