# stdlib
import copy
import hashlib
from typing import Any, Callable, Dict, List

# third party
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# synthcity absolute
import synthcity.logger as log
from synthcity.plugins.core.models.survival_analysis.metrics import (
    evaluate_brier_score,
    evaluate_c_index,
    generate_score,
    print_score,
)
from synthcity.utils.optimizer import (
    EarlyStoppingExceeded,
    ParamRepeatPruner,
    create_study,
)


def _trial_params(estimator_name: str, trial: optuna.Trial, param_space: List) -> Dict:
    out = {}

    for param in param_space:
        key = f"{estimator_name}::{param.name}"
        if hasattr(param, "choices"):
            out[param.name] = trial.suggest_categorical(key, choices=param.choices)
        elif hasattr(param, "step"):
            out[param.name] = trial.suggest_int(key, param.low, param.high, param.step)
        else:
            out[param.name] = trial.suggest_float(key, param.low, param.high)

    return out


def _normalize_params(estimator_name: str, args: dict) -> dict:
    prefix = f"{estimator_name}::"

    out = {}

    for key in args:
        norm = key.split(prefix)[-1]
        out[norm] = args[key]

    return out


def _search_objective_meta(
    estimator: Any,
    static: np.ndarray,
    temporal: np.ndarray,
    observation_times: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    time_horizons: List,
    pruner: ParamRepeatPruner,
    n_folds: int = 3,
) -> Callable:
    def objective(trial: optuna.Trial) -> float:
        args = _trial_params(estimator.name(), trial, estimator.hyperparameter_space())
        pruner.check_trial(trial)
        try:
            model = estimator(n_iter=10, **args)
            raw_score = evaluate_ts_survival_model(
                model, static, temporal, observation_times, T, Y, time_horizons
            )
        except BaseException as e:
            log.error(f"model search failed {e}")
            return 0

        score = raw_score["clf"]["c_index"][0] - raw_score["clf"]["brier_score"][0]
        pruner.report_score(score)

        return score

    return objective


def search_hyperparams(
    estimator: Any,
    static: np.ndarray,
    temporal: np.ndarray,
    observation_times: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    time_horizons: List,
    n_folds: int = 3,
    metrics: List[str] = ["c_index", "brier_score"],
    random_state: int = 0,
    pretrained: bool = False,
    n_trials: int = 50,
    timeout: int = 100,
) -> dict:
    temporal_total = 0
    for item in temporal:
        temporal_total += item.sum()
    data = str((static.sum(), temporal_total, T.sum(), Y.sum())).encode("utf-8")
    data_str = hashlib.sha256(data).hexdigest()
    study, pruner = create_study(
        study_name=f"ts_survival_eval_{data_str}_{estimator.name()}",
        direction="maximize",
    )

    try:
        study.optimize(
            _search_objective_meta(
                estimator,
                static,
                temporal,
                observation_times,
                T,
                Y,
                time_horizons,
                pruner,
                n_folds=n_folds,
            ),
            n_trials=n_trials,
            timeout=timeout,
        )

    except EarlyStoppingExceeded:
        pass
    except BaseException as e:
        log.error(f"model {estimator.name()} failed: {e}")
        return {}

    score = study.best_trial.value
    args = _normalize_params(estimator.name(), study.best_trial.params)
    log.info(
        f"[select_ts_survival] model = {estimator.name()} score = {score} args = {args}"
    )

    return args


def evaluate_ts_survival_model(
    estimator: Any,
    static: np.ndarray,
    temporal: np.ndarray,
    observation_times: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    time_horizons: List,
    n_folds: int = 3,
    metrics: List[str] = ["c_index", "brier_score"],
    random_state: int = 0,
    pretrained: bool = False,
) -> Dict:
    """Helper for evaluating survival analysis tasks.

    Args:
        model_name: str
            The model to evaluate
        model_args: dict
            The model args to use
        static: np.ndarray
            The static covariates
        temporal: np.ndarray
            The temporal covariates
        observation_times: np.ndarray
            The temporal points
        T: np.ndarray
            time to event
        Y: np.ndarray
            event or censored
        time_horizons: list
            Horizons where to evaluate the performance.
        n_folds: int
            Number of folds for cross validation
        metrics: list
            Available metrics: "c_index", "brier_score"
        random_state: int
            Random random_state
        pretrained: bool
            If the estimator was trained or not
    """

    supported_metrics = ["c_index", "brier_score"]
    results = {}

    static = np.asarray(static)
    temporal = np.asarray(temporal, dtype=object)
    observation_times = np.asarray(observation_times, dtype=object)
    T = np.asarray(T)
    Y = np.asarray(Y)

    for metric in metrics:
        if metric not in supported_metrics:
            raise ValueError(f"Metric {metric} not supported")

        results[metric] = np.zeros(n_folds)

    def _get_surv_metrics(
        cv_idx: int,
        static_train: np.ndarray,
        static_test: np.ndarray,
        temporal_train: np.ndarray,
        temporal_test: np.ndarray,
        observation_times_train: np.ndarray,
        observation_times_test: np.ndarray,
        T_train: np.ndarray,
        T_test: np.ndarray,
        Y_train: np.ndarray,
        Y_test: np.ndarray,
        time_horizons: list,
    ) -> tuple:
        train_max = T_train.max()
        T_test[T_test > train_max] = train_max

        if pretrained:
            model = estimator[cv_idx]
        else:
            model = copy.deepcopy(estimator)

            model.fit(
                static_train, temporal_train, observation_times_train, T_train, Y_train
            )
        try:
            pred = model.predict(
                static_test, temporal_test, observation_times_test, time_horizons
            ).to_numpy()
        except BaseException as e:
            raise e

        c_index = 0.0
        brier_score = 0.0

        for k in range(len(time_horizons)):
            eval_horizon = min(time_horizons[k], np.max(T_test) - 1)

            def get_score(fn: Callable) -> float:
                return fn(
                    T_train,
                    Y_train,
                    pred[:, k],
                    T_test,
                    Y_test,
                    eval_horizon,
                ) / (len(time_horizons))

            c_index += get_score(evaluate_c_index)
            brier_score += get_score(evaluate_brier_score)

        return c_index, brier_score

    if n_folds == 1:
        cv_idx = 0
        (
            static_train,
            static_test,
            temporal_train,
            temporal_test,
            observation_times_train,
            observation_times_test,
            T_train,
            T_test,
            Y_train,
            Y_test,
        ) = train_test_split(
            static, temporal, observation_times, T, Y, random_state=random_state
        )
        local_time_horizons = [t for t in time_horizons if t > np.min(T_test)]

        c_index, brier_score = _get_surv_metrics(
            cv_idx,
            static_train,
            static_test,
            temporal_train,
            temporal_test,
            observation_times_train,
            observation_times_test,
            T_train,
            T_test,
            Y_train,
            Y_test,
            local_time_horizons,
        )
        for metric in metrics:
            if metric == "c_index":
                results[metric][cv_idx] = c_index
            elif metric == "brier_score":
                results[metric][cv_idx] = brier_score

    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        cv_idx = 0
        for train_index, test_index in skf.split(temporal, Y):
            static_train = static[train_index]
            temporal_train = temporal[train_index]
            observation_times_train = observation_times[train_index]
            Y_train = Y[train_index]
            T_train = T[train_index]

            static_test = static[test_index]
            temporal_test = temporal[test_index]
            observation_times_test = observation_times[test_index]
            Y_test = Y[test_index]
            T_test = T[test_index]

            local_time_horizons = [t for t in time_horizons if t > np.min(T_test)]

            c_index, brier_score = _get_surv_metrics(
                cv_idx,
                static_train,
                static_test,
                temporal_train,
                temporal_test,
                observation_times_train,
                observation_times_test,
                T_train,
                T_test,
                Y_train,
                Y_test,
                local_time_horizons,
            )
            for metric in metrics:
                if metric == "c_index":
                    results[metric][cv_idx] = c_index
                elif metric == "brier_score":
                    results[metric][cv_idx] = brier_score

            cv_idx += 1

    output: dict = {
        "clf": {},
        "str": {},
    }

    for metric in metrics:
        output["clf"][metric] = generate_score(results[metric])
        output["str"][metric] = print_score(output["clf"][metric])

    return output


def evaluate_ts_classification(
    estimator: Any,
    static: np.ndarray,
    temporal: np.ndarray,
    observation_times: np.ndarray,
    Y: np.ndarray,
    n_folds: int = 3,
    metrics: List[str] = ["aucroc"],
    random_state: int = 0,
    pretrained: bool = False,
) -> Dict:
    results: Dict[str, list] = {
        "aucroc": [],
    }

    static = np.asarray(static)
    temporal = np.asarray(temporal)
    observation_times = np.asarray(observation_times)
    Y = np.asarray(Y)

    def _get_metrics(
        cv_idx: int,
        static_train: np.ndarray,
        static_test: np.ndarray,
        temporal_train: np.ndarray,
        temporal_test: np.ndarray,
        observation_times_train: np.ndarray,
        observation_times_test: np.ndarray,
        Y_train: np.ndarray,
        Y_test: np.ndarray,
    ) -> tuple:
        if pretrained:
            model = estimator[cv_idx]
        else:
            model = copy.deepcopy(estimator)

            model.fit(static_train, temporal_train, observation_times_train, Y_train)
        pred = model.predict(static_test, temporal_test, observation_times_test)

        return roc_auc_score(Y_test, pred)

    if n_folds == 1:
        cv_idx = 0
        (
            static_train,
            static_test,
            temporal_train,
            temporal_test,
            observation_times_train,
            observation_times_test,
            Y_train,
            Y_test,
        ) = train_test_split(
            static, temporal, observation_times, Y, random_state=random_state
        )

        aucroc = _get_metrics(
            cv_idx,
            static_train,
            static_test,
            temporal_train,
            temporal_test,
            observation_times_train,
            observation_times_test,
            Y_train,
            Y_test,
        )
        results["aucroc"] = [aucroc]
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        cv_idx = 0
        for train_index, test_index in skf.split(temporal, Y):
            static_train = static[train_index]
            temporal_train = temporal[train_index]
            observation_times_train = observation_times[train_index]
            Y_train = Y[train_index]

            static_test = static[test_index]
            temporal_test = temporal[test_index]
            observation_times_test = observation_times[test_index]
            Y_test = Y[test_index]

            aucroc = _get_metrics(
                cv_idx,
                static_train,
                static_test,
                temporal_train,
                temporal_test,
                observation_times_train,
                observation_times_test,
                Y_train,
                Y_test,
            )
            results["aucroc"].append(aucroc)

            cv_idx += 1

    output: dict = {
        "clf": {},
        "str": {},
    }

    for metric in metrics:
        output["clf"][metric] = generate_score(results[metric])
        output["str"][metric] = print_score(output["clf"][metric])

    return output
