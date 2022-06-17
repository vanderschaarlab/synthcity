# stdlib
import copy
from typing import Any, Callable, Dict, List

# third party
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

# synthcity absolute
from synthcity.plugins.core.models.survival_analysis.metrics import (
    evaluate_brier_score,
    evaluate_c_index,
    generate_score,
    print_score,
)


def search_hyperparams(
    estimator: Any,
    static: np.ndarray,
    temporal: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    time_horizons: List,
    n_folds: int = 3,
    metrics: List[str] = ["c_index", "brier_score"],
    seed: int = 0,
    pretrained: bool = False,
    scenarios: int = 10,
) -> dict:
    params_list = []
    for t in range(scenarios):
        params_list.append(estimator.sample_hyperparameters())

    best_c = 0
    best_params = {}
    for param in params_list:
        model = estimator(n_iter=1, **param)
        score = evaluate_ts_survival_model(model, static, temporal, T, Y, time_horizons)

        if score["clf"]["c_index"][0] > best_c:
            best_c = score["clf"]["c_index"][0]
            best_params = param

    return best_params


def evaluate_ts_survival_model(
    estimator: Any,
    static: np.ndarray,
    temporal: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    time_horizons: List,
    n_folds: int = 3,
    metrics: List[str] = ["c_index", "brier_score"],
    seed: int = 0,
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
        seed: int
            Random seed
        pretrained: bool
            If the estimator was trained or not
    """

    supported_metrics = ["c_index", "brier_score"]
    results = {}

    static = np.asarray(static)
    temporal = np.asarray(temporal)
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

            model.fit(static_train, temporal_train, T_train, Y_train)
        try:
            pred = model.predict(static_test, temporal_test, time_horizons).to_numpy()
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
            T_train,
            T_test,
            Y_train,
            Y_test,
        ) = train_test_split(static, temporal, T, Y, random_state=seed)
        local_time_horizons = [t for t in time_horizons if t > np.min(T_test)]

        c_index, brier_score = _get_surv_metrics(
            cv_idx,
            static_train,
            static_test,
            temporal_train,
            temporal_test,
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
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        cv_idx = 0
        for train_index, test_index in skf.split(temporal, Y):
            static_train = static[train_index]
            temporal_train = temporal[train_index]
            Y_train = Y[train_index]
            T_train = T[train_index]

            static_test = static[test_index]
            temporal_test = temporal[test_index]
            Y_test = Y[test_index]
            T_test = T[test_index]

            local_time_horizons = [t for t in time_horizons if t > np.min(T_test)]

            c_index, brier_score = _get_surv_metrics(
                cv_idx,
                static_train,
                static_test,
                temporal_train,
                temporal_test,
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
