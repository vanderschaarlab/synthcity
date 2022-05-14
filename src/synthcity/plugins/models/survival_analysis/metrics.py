# stdlib
from typing import Tuple

# third party
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy.integrate import trapz
from xgbse.converters import hazard_to_survival
from xgbse.extrapolation import extrapolate_constant_risk
from xgbse.non_parametric import _get_conditional_probs_from_survival

# synthcity absolute
from synthcity.plugins.models.survival_analysis.third_party.metrics import (
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kmf = KaplanMeierFitter().fit(T, E)
    surv_fn = kmf.survival_function_.T.reset_index(drop=True)
    if len(surv_fn.columns) < 2:
        raise RuntimeError("invalid survival functin for extrapolation")

    hazards = _get_conditional_probs_from_survival(surv_fn)
    constant_hazard = hazards.values[:, -1:].mean(axis=1)[0]

    last_point = T.max()
    ext_surv_fn = surv_fn.copy()

    while ext_surv_fn[ext_surv_fn.columns[-1]].values[0] > 0.01:
        ext_surv_fn = extrapolate_constant_risk(ext_surv_fn, last_point + T.max(), 1)
        last_point += T.max()

    return surv_fn, ext_surv_fn, hazards, constant_hazard


def nonparametric_distance(
    real: Tuple[np.ndarray, np.ndarray],
    syn: Tuple[np.ndarray, np.ndarray],
) -> Tuple:
    real_T, real_E = real
    syn_T, syn_E = syn

    time_points = real_T.append(syn_T).unique()
    time_points.sort()

    Tmax = max(real_T.max(), syn_T.max())
    opt = []
    abs_opt = []

    real_surv, _, real_hazards, real_constant_hazard = km_survival_function(
        real_T, real_E
    )
    syn_surv, _, syn_hazards, syn_constant_hazard = km_survival_function(syn_T, syn_E)

    for t in time_points:
        if t not in real_hazards:
            real_hazards[t] = real_constant_hazard
        if t not in syn_hazards:
            syn_hazards[t] = syn_constant_hazard

    real_surv = hazard_to_survival(real_hazards)
    syn_surv = hazard_to_survival(syn_hazards)

    for idx, t in enumerate(time_points):
        t_real = real_surv.values[0][idx]
        t_syn = syn_surv.values[0][idx]

        opt.append(t_syn - t_real)
        abs_opt.append(np.abs(t_syn - t_real))

    auc_opt = trapz(opt, time_points) / Tmax
    auc_abs_opt = trapz(abs_opt, time_points) / Tmax
    sightedness = (real_T.max() - syn_T.max()) / Tmax

    return auc_opt, auc_abs_opt, sightedness


def generate_score(metric: np.ndarray) -> Tuple[float, float]:
    percentile_val = 1.96
    return (np.mean(metric), percentile_val * np.std(metric) / np.sqrt(len(metric)))


def print_score(score: Tuple[float, float]) -> str:
    return str(round(score[0], 4)) + " +/- " + str(round(score[1], 4))
