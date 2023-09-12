# stdlib
import copy
from typing import Any, Callable, Dict, List

# third party
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# synthcity absolute
from synthcity.plugins.core.models.survival_analysis.loader import (
    generate_dataset_for_horizon,
)
from synthcity.plugins.core.models.survival_analysis.metrics import (
    evaluate_brier_score,
    evaluate_c_index,
    generate_score,
    print_score,
)
from synthcity.utils.dataframe import constant_columns


def evaluate_survival_model(
    estimator: Any,
    X: pd.DataFrame,
    T: pd.DataFrame,
    Y: pd.DataFrame,
    time_horizons: List,
    n_folds: int = 3,
    metrics: List[str] = ["c_index", "brier_score", "aucroc"],
    random_state: int = 0,
    pretrained: bool = False,
) -> Dict:
    """Helper for evaluating survival analysis tasks.

    Args:
        model_name: str
            The model to evaluate
        model_args: dict
            The model args to use
        X: DataFrame
            The covariates
        T: Series
            time to event
        Y: Series
            event or censored
        time_horizons: list
            Horizons where to evaluate the performance.
        n_folds: int
            Number of folds for cross validation
        metrics: list
            Available metrics: "c_index", "brier_score", "aucroc"
        random_state: int
            Random seed
        pretrained: bool
            If the estimator was trained or not
    """

    supported_metrics = ["c_index", "brier_score", "aucroc"]
    results = {}

    for metric in metrics:
        if metric not in supported_metrics:
            raise ValueError(f"Metric {metric} not supported")

        results[metric] = np.zeros(n_folds)

    def _get_surv_metrics(
        cv_idx: int,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        T_train: pd.DataFrame,
        T_test: pd.DataFrame,
        Y_train: pd.DataFrame,
        Y_test: pd.DataFrame,
        time_horizons: list,
    ) -> tuple:
        train_max = T_train.max()
        T_test[T_test > train_max] = train_max

        if pretrained:
            model = estimator[cv_idx]
        else:
            model = copy.deepcopy(estimator)

            constant_cols = constant_columns(X_train)
            X_train = X_train.drop(columns=constant_cols)
            X_test = X_test.drop(columns=constant_cols)

            model.fit(X_train, T_train, Y_train)

        try:
            pred = model.predict(X_test, time_horizons).to_numpy()
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

    def _get_clf_metrics(
        cv_idx: int,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        T_train: pd.DataFrame,
        T_test: pd.DataFrame,
        Y_train: pd.DataFrame,
        Y_test: pd.DataFrame,
        time_horizons: list,
    ) -> float:
        cv_idx = 0

        train_max = T_train.max()
        T_test[T_test > train_max] = train_max

        if pretrained:
            model = estimator[cv_idx]
        else:
            model = copy.deepcopy(estimator)

            constant_cols = constant_columns(X_train)
            X_train = X_train.drop(columns=constant_cols)
            X_test = X_test.drop(columns=constant_cols)

            model.fit(X_train, T_train, Y_train)

        try:
            pred = model.predict(X_test, time_horizons).to_numpy()
        except BaseException as e:
            raise e

        local_preds = pd.DataFrame(pred[:, k]).squeeze()

        return roc_auc_score(Y_test, local_preds) / (len(time_horizons))

    if n_folds == 1:
        cv_idx = 0
        X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
            X, T, Y, random_state=random_state
        )
        local_time_horizons = [t for t in time_horizons if t > np.min(T_test)]

        c_index, brier_score = _get_surv_metrics(
            cv_idx,
            X_train,
            X_test,
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

        if "aucroc" in metrics:
            for k in range(len(time_horizons)):
                cv_idx = 0

                X_horizon, T_horizon, Y_horizon = generate_dataset_for_horizon(
                    X, T, Y, time_horizons[k]
                )
                X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
                    X_horizon, T_horizon, Y_horizon, random_state=random_state
                )

                metric = "aucroc"

                results[metric][cv_idx] += _get_clf_metrics(
                    cv_idx,
                    X_train,
                    X_test,
                    T_train,
                    T_test,
                    Y_train,
                    Y_test,
                    local_time_horizons,
                )

    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        cv_idx = 0
        for train_index, test_index in skf.split(X, Y):

            X_train = X.loc[X.index[train_index]]
            Y_train = Y.loc[Y.index[train_index]]
            T_train = T.loc[T.index[train_index]]
            X_test = X.loc[X.index[test_index]]
            Y_test = Y.loc[Y.index[test_index]]
            T_test = T.loc[T.index[test_index]]

            local_time_horizons = [t for t in time_horizons if t > np.min(T_test)]

            c_index, brier_score = _get_surv_metrics(
                cv_idx,
                X_train,
                X_test,
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

        if "aucroc" in metrics:
            for k in range(len(time_horizons)):
                cv_idx = 0

                X_horizon, T_horizon, Y_horizon = generate_dataset_for_horizon(
                    X, T, Y, time_horizons[k]
                )
                for train_index, test_index in skf.split(X_horizon, Y_horizon):

                    X_train = X_horizon.loc[X_horizon.index[train_index]]
                    Y_train = Y_horizon.loc[Y_horizon.index[train_index]]
                    T_train = T_horizon.loc[T_horizon.index[train_index]]
                    X_test = X_horizon.loc[X_horizon.index[test_index]]
                    Y_test = Y_horizon.loc[Y_horizon.index[test_index]]
                    T_test = T_horizon.loc[T_horizon.index[test_index]]

                    metric = "aucroc"

                    results[metric][cv_idx] += _get_clf_metrics(
                        cv_idx,
                        X_train,
                        X_test,
                        T_train,
                        T_test,
                        Y_train,
                        Y_test,
                        local_time_horizons,
                    )

                    cv_idx += 1

    output: dict = {
        "clf": {},
        "str": {},
    }

    for metric in metrics:
        output["clf"][metric] = generate_score(results[metric])
        output["str"][metric] = print_score(output["clf"][metric])

    return output
