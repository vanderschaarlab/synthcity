# stdlib
from time import time
from typing import Any, Callable, Dict, List

# third party
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# synthcity absolute
import synthcity.logger as log
from synthcity.plugins.core.models.time_to_event.loader import get_model_template
from synthcity.plugins.core.models.time_to_event.metrics import (
    c_index,
    expected_time_error,
)
from synthcity.utils.optimizer import (
    EarlyStoppingExceeded,
    ParamRepeatPruner,
    create_study,
)
from synthcity.utils.serialization import dataframe_hash


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
    random_state: int = 0,
) -> tuple:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

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
        except BaseException as e:
            log.error(f"fold failed {e}")
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


def _trial_params(trial: optuna.Trial, param_space: List) -> Dict:
    out = {}

    for param in param_space:
        if hasattr(param, "choices"):
            out[param.name] = trial.suggest_categorical(
                param.name, choices=param.choices
            )
        elif hasattr(param, "step"):
            out[param.name] = trial.suggest_int(
                param.name, param.low, param.high, param.step
            )
        else:
            out[param.name] = trial.suggest_float(param.name, param.low, param.high)

    return out


def objective_meta(
    model_name: str,
    X: pd.DataFrame,
    T: pd.DataFrame,
    E: pd.DataFrame,
    metric: str,
    pruner: ParamRepeatPruner,
    n_folds: int = 3,
) -> Callable:
    def objective(trial: optuna.Trial) -> float:
        template = get_model_template(model_name)
        args = _trial_params(trial, template.hyperparameter_space())
        pruner.check_trial(trial)
        try:
            full_score, _ = evaluate_model(model_name, args, X, T, E, n_folds=n_folds)
        except BaseException as e:
            log.error(f"model search failed {e}")
            return 0

        score = full_score[metric][0]
        pruner.report_score(score)

        return score

    return objective


def select_uncensoring_model(
    X: pd.DataFrame,
    T: pd.DataFrame,
    E: pd.DataFrame,
    random_states: List[str] = [
        "weibull_aft",
        "cox_ph",
        "random_survival_forest",
        "survival_xgboost",
        "deephit",
        "tenn",
        "date",
    ],
    n_folds: int = 2,
    n_trials: int = 10,
    timeout: int = 120,
    random_state: int = 0,
) -> Any:
    metric = "c_index_ood"

    df_hash = dataframe_hash(pd.concat([X, T, E], axis=1))

    log.info(f"Evaluate uncensoring for {df_hash}")

    candidate = {
        "model": "tenn",
        "args": {},
        "score": 0,
    }
    for model in random_states:
        start = time()
        study, pruner = create_study(
            study_name=f"uncensoring_{df_hash}_{model}_{metric}",
            direction="maximize",
        )

        try:
            study.optimize(
                objective_meta(model, X, T, E, metric, pruner, n_folds=n_folds),
                n_trials=n_trials,
                timeout=timeout,
            )
            if study.best_trial is None or study.best_trial.value is None:
                continue

        except EarlyStoppingExceeded:
            pass
        except BaseException as e:
            log.error(f"model {model} failed: {e}")
            continue

        score = study.best_trial.value

        log.info(
            f"[select_uncensoring_model] model = {model} {metric} = {score} duration = {round(time() - start, 4)} s"
        )
        if score > candidate["score"]:
            candidate = {
                "model": model,
                "args": study.best_trial.params,
                "score": score,
            }

    model_template = get_model_template(str(candidate["model"]))

    return model_template(**candidate["args"])
