# stdlib
from typing import List, Tuple

# third party
import numpy as np
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.models.survival_analysis.metrics import (
    evaluate_brier_score,
    evaluate_c_index,
)


def flatten(T: List[np.ndarray], E: List[np.ndarray]) -> Tuple:
    T_flat = []
    E_flat = []

    for idx, item in enumerate(T):
        T_flat.extend(item)
        E_flat.extend(E[idx])
    return T_flat, E_flat


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_c_index_ts(
    T_train: np.ndarray,
    Y_train: np.ndarray,
    Prediction: np.ndarray,
    T_test: np.ndarray,
    Y_test: np.ndarray,
    Time: float,
) -> float:
    """Helper for evaluating the C-INDEX metric."""
    T_train_flat, E_train_flat = flatten(T_train, Y_train)
    T_test_flat, E_test_flat = flatten(T_test, Y_test)

    assert len(T_test_flat) == len(Prediction)

    return evaluate_c_index(
        T_train_flat, E_train_flat, Prediction, T_test_flat, E_test_flat, Time
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_brier_score_ts(
    T_train: np.ndarray,
    Y_train: np.ndarray,
    Prediction: np.ndarray,
    T_test: np.ndarray,
    Y_test: np.ndarray,
    Time: float,
) -> float:
    """Helper for evaluating the Brier score."""
    T_train_flat, E_train_flat = flatten(T_train, Y_train)
    T_test_flat, E_test_flat = flatten(T_test, Y_test)

    assert len(T_test_flat) == len(Prediction)

    return evaluate_brier_score(
        T_train_flat, E_train_flat, Prediction, T_test_flat, E_test_flat, Time
    )


def generate_score(metric: np.ndarray) -> Tuple[float, float]:
    percentile_val = 1.96
    return (np.mean(metric), percentile_val * np.std(metric) / np.sqrt(len(metric)))


def print_score(score: Tuple[float, float]) -> str:
    return str(round(score[0], 4)) + " +/- " + str(round(score[1], 4))
