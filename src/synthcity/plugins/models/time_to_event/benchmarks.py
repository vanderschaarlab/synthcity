# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# synthcity absolute
from synthcity.plugins.models.time_to_event.loader import get_model_template
from synthcity.plugins.models.time_to_event.metrics import c_index, expected_time_error


def generate_score(metric: np.ndarray) -> tuple:
    percentile_val = 1.96
    score = (np.mean(metric), percentile_val * np.std(metric) / np.sqrt(len(metric)))

    return round(score[0], 4), round(score[1], 4)


def generate_score_str(metric: np.ndarray) -> str:
    mean, std = generate_score(metric)
    return str(mean) + " +/- " + str(std)


def evaluate_model(
    model_name: str,
    model_args: dict,
    X: pd.DataFrame,
    T: pd.DataFrame,
    E: pd.DataFrame,
    n_folds: int = 3,
    seed: int = 0,
) -> tuple:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    local_results: dict = {
        "te_err_l1_ood": [],
        "te_err_l2_ood": [],
        "c_index_ood": [],
        "te_err_l1_id": [],
        "te_err_l2_id": [],
        "c_index_id": [],
    }

    model_template = get_model_template(model_name)

    for train_index, test_index in skf.split(X, E):
        X_train = X.loc[X.index[train_index]]
        E_train = E.loc[E.index[train_index]]
        T_train = T.loc[T.index[train_index]]

        X_test = X.loc[X.index[test_index]]
        E_test = E.loc[E.index[test_index]]
        T_test = T.loc[T.index[test_index]]

        model = model_template(**model_args)

        try:
            model.fit(X_train, T_train, E_train)
            ood_preds = model.predict(X_test)
            id_preds = model.predict(X_train)

        except BaseException:
            continue

        local_results["te_err_l1_ood"].append(
            expected_time_error(T_test, E_test, ood_preds, metric="l1")
        )
        local_results["te_err_l2_ood"].append(
            expected_time_error(T_test, E_test, ood_preds, metric="l2")
        )
        local_results["c_index_ood"].append(c_index(T_test, E_test, ood_preds))

        local_results["te_err_l1_id"].append(
            expected_time_error(T_train, E_train, id_preds, metric="l1")
        )
        local_results["te_err_l2_id"].append(
            expected_time_error(T_train, E_train, id_preds, metric="l2")
        )
        local_results["c_index_id"].append(c_index(T_train, E_train, id_preds))

    output = {}
    output_str = {}
    for metric in local_results:
        output[metric] = generate_score(local_results[metric])
        output_str[metric] = generate_score(local_results[metric])

    return output, output_str


def select_uncensoring_model(
    X: pd.DataFrame,
    T: pd.DataFrame,
    E: pd.DataFrame,
    seeds: List[str] = [
        "weibull_aft",
        "cox_ph",
        "random_survival_forest",
        "survival_xgboost",
        "deephit",
        "tenn",
        "date",
    ],
    n_folds: int = 3,
    seed: int = 0,
) -> Any:
    metric = "c_index_ood"
    candidate = {
        "model": "tenn",
        "args": {},
        "score": 0,
    }
    for model in seeds:
        full_score, _ = evaluate_model(model, {}, X, T, E)
        if metric not in full_score:
            raise RuntimeError(f"Metric {metric} not found")
        score = full_score[metric][0]

        if score > candidate["score"]:
            candidate = {
                "model": model,
                "args": {},
                "score": score,
            }

    model_template = get_model_template(str(candidate["model"]))

    return model_template(**candidate["args"])
