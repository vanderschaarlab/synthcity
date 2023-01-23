# stdlib
import copy
import platform
from typing import Any, Dict, Tuple

# third party
import numpy as np
import pandas as pd
import shap
from pydantic import validate_arguments
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

# synthcity absolute
import synthcity.logger as log
from synthcity.metrics._utils import evaluate_auc
from synthcity.metrics.core import MetricEvaluator
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.models.mlp import MLP
from synthcity.plugins.core.models.survival_analysis import (
    CoxPHSurvivalAnalysis,
    DeephitSurvivalAnalysis,
    XGBSurvivalAnalysis,
    evaluate_survival_model,
)
from synthcity.plugins.core.models.time_series_survival.benchmarks import (
    evaluate_ts_survival_model,
)
from synthcity.plugins.core.models.time_series_survival.ts_surv_coxph import (
    CoxTimeSeriesSurvival,
)
from synthcity.plugins.core.models.time_series_survival.ts_surv_dynamic_deephit import (
    DynamicDeephitTimeSeriesSurvival,
)
from synthcity.plugins.core.models.time_series_survival.ts_surv_xgb import (
    XGBTimeSeriesSurvival,
)
from synthcity.plugins.core.models.ts_model import TimeSeriesModel
from synthcity.utils.serialization import load_from_file, save_to_file


class PerformanceEvaluator(MetricEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_performance.PerformanceEvaluator
        :parts: 1

    Evaluating synthetic data based on downstream performance.

    This implements the train-on-synthetic test-on-real methodology for evaluation.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        return "performance"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def _evaluate_performance_classification(
        self,
        model: Any,
        model_args: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        """
        Evaluate a classification task.

        Returns: AUCROC.
            1 means perfect predictions.
            0 means only incorrect predictions.
        """
        labels = list(y_train) + list(y_test)

        X_test_df = pd.DataFrame(X_test)
        y_test_df = pd.Series(y_test, index=X_test_df.index)
        for v in np.unique(labels):
            if v not in list(y_train):
                X_test_df = X_test_df[y_test_df != v]
                y_test_df = y_test_df[y_test_df != v]

        X_test = np.asarray(X_test_df)
        y_test = np.asarray(y_test_df)
        labels = list(y_train) + list(y_test)

        if len(y_test) == 0:
            return 0

        encoder = LabelEncoder().fit(labels)
        enc_y_train = encoder.transform(y_train)
        if "n_units_out" in model_args:
            model_args["n_units_out"] = len(np.unique(y_train))
        try:
            enc_y_test = encoder.transform(y_test)
            estimator = model(**model_args).fit(X_train, enc_y_train)
            y_pred = estimator.predict_proba(X_test)
            score, _ = evaluate_auc(enc_y_test, y_pred)
        except BaseException as e:
            log.error(f"classifier evaluation failed {e}.")
            score = 0

        return score

    def _evaluate_performance_regression(
        self,
        model: Any,
        model_args: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        """
        Evaluate a regression task.

        Returns: 1/ (1 + RMSE).
            0 means perfect predictions.
            The lower the negative value, the bigger the error in the predictions.
        """

        X_train = np.asarray(X_test)
        X_test = np.asarray(X_test)
        y_train = np.asarray(y_test)
        y_test = np.asarray(y_test)

        try:
            estimator = model(**model_args).fit(X_train, y_train)
            y_pred = estimator.predict(X_test)

            score = r2_score(y_test, y_pred)
        except BaseException as e:
            log.error(f"regression evaluation failed {e}")
            score = -1

        return score

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate_standard_performance(
        self,
        clf_model: Any,
        clf_args: Dict,
        regression_model: Any,
        regression_args: Any,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        """Train a classifier or regressor on the synthetic data and evaluate the performance on real test data. Returns the average performance discrepancy between training on real data vs on synthetic data.

        Returns:
            gt and syn performance scores
        """
        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{X_gt.hash()}_{X_syn.hash()}_{self._reduction}_{platform.python_version()}_{platform.python_version()}.bkp"
        )
        if self.use_cache(cache_file):
            return load_from_file(cache_file)

        id_X_gt, id_y_gt = X_gt.train().unpack()
        ood_X_gt, ood_y_gt = X_gt.test().unpack()
        iter_X_syn, iter_y_syn = X_syn.unpack()

        if len(id_y_gt.unique()) < 5:
            eval_cbk = self._evaluate_performance_classification
            skf = StratifiedKFold(
                n_splits=self._n_folds, shuffle=True, random_state=self._random_state
            )
            model = clf_model
            model_args = clf_args
        else:
            eval_cbk = self._evaluate_performance_regression
            model = regression_model
            model_args = regression_args
            skf = KFold(
                n_splits=self._n_folds, shuffle=True, random_state=self._random_state
            )

        real_scores = []
        syn_scores_id = []
        syn_scores_ood = []

        for train_idx, test_idx in skf.split(id_X_gt, id_y_gt):
            train_data = np.asarray(id_X_gt.loc[train_idx])
            test_data = np.asarray(id_X_gt.loc[test_idx])
            train_labels = np.asarray(id_y_gt.loc[train_idx])
            test_labels = np.asarray(id_y_gt.loc[test_idx])

            real_score = eval_cbk(
                model, model_args, train_data, train_labels, test_data, test_labels
            )
            synth_score_id = eval_cbk(
                model, model_args, iter_X_syn, iter_y_syn, test_data, test_labels
            )
            synth_score_ood = eval_cbk(
                model, model_args, iter_X_syn, iter_y_syn, ood_X_gt, ood_y_gt
            )

            real_scores.append(real_score)
            syn_scores_id.append(synth_score_id)
            syn_scores_ood.append(synth_score_ood)

        results = {
            "gt": float(self.reduction()(real_scores)),
            "syn_id": float(self.reduction()(syn_scores_id)),
            "syn_ood": float(self.reduction()(syn_scores_ood)),
        }

        save_to_file(cache_file, results)

        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate_survival_model(
        self,
        model: Any,
        args: Dict,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        """Train a survival model on the synthetic data and evaluate the performance on real test data. Returns the average performance discrepancy between training on real data vs on synthetic data.

        Returns:
            gt and syn performance scores
        """
        if X_gt.type() != "survival_analysis" or X_syn.type() != "survival_analysis":
            raise ValueError(
                f"Invalid data types. gt = {X_gt.type()} syn = {X_syn.type()}"
            )

        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{X_gt.hash()}_{X_syn.hash()}_{self._reduction}_{platform.python_version()}_{platform.python_version()}.bkp"
        )
        if self.use_cache(cache_file):
            return load_from_file(cache_file)

        info = X_gt.info()
        time_horizons = info["time_horizons"]

        id_X_gt, id_T_gt, id_E_gt = X_gt.train().unpack()
        ood_X_gt, ood_T_gt, ood_E_gt = X_gt.test().unpack()
        iter_X_syn, iter_T_syn, iter_E_syn = X_syn.unpack()

        predictor_gt = model(**args)
        log.info(
            f" Performance eval for df hash = {X_gt.train().hash()} ood hash = {X_gt.test().hash()}"
        )
        score_gt = evaluate_survival_model(
            predictor_gt,
            id_X_gt,
            id_T_gt,
            id_E_gt,
            metrics=["c_index", "brier_score"],
            n_folds=self._n_folds,
            time_horizons=time_horizons,
        )["clf"]

        log.info(f"Baseline performance score: {score_gt}")

        predictor_syn = model(**args)

        fail_score = {
            "c_index": (0, 0),
            "brier_score": (1, 0),
        }
        try:
            predictor_syn.fit(iter_X_syn, iter_T_syn, iter_E_syn)
            score_syn_id = evaluate_survival_model(
                [predictor_syn] * self._n_folds,
                id_X_gt,
                id_T_gt,
                id_E_gt,
                metrics=["c_index", "brier_score"],
                n_folds=self._n_folds,
                time_horizons=time_horizons,
                pretrained=True,
            )["clf"]
        except BaseException as e:
            log.error(
                f"Failed to evaluate synthetic ID performance. {model.name()}: {e}"
            )
            score_syn_id = fail_score

        log.info(f"Synthetic ID performance score: {score_syn_id}")

        try:
            predictor_syn.fit(iter_X_syn, iter_T_syn, iter_E_syn)
            score_syn_ood = evaluate_survival_model(
                [predictor_syn] * self._n_folds,
                ood_X_gt,
                ood_T_gt,
                ood_E_gt,
                metrics=["c_index", "brier_score"],
                n_folds=self._n_folds,
                time_horizons=time_horizons,
                pretrained=True,
            )["clf"]
        except BaseException as e:
            log.error(
                f"Failed to evaluate synthetic OOD performance. {model.name()}: {e}"
            )
            score_syn_ood = fail_score

        log.info(f"Synthetic OOD performance score: {score_syn_ood}")

        results = {
            "gt.c_index": float(score_gt["c_index"][0]),
            "gt.brier_score": float(score_gt["brier_score"][0]),
            "syn_id.c_index": float(score_syn_id["c_index"][0]),
            "syn_id.brier_score": float(score_syn_id["brier_score"][0]),
            "syn_ood.c_index": float(score_syn_ood["c_index"][0]),
            "syn_ood.brier_score": float(score_syn_ood["brier_score"][0]),
        }

        save_to_file(cache_file, results)

        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate_time_series_performance(
        self,
        model: Any,
        model_args: Any,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        """Train a regressor on the time series data and evaluate the performance on real test data. Returns the average performance discrepancy between training on real data vs on synthetic data.

        Returns:
            gt and syn performance scores
        """
        if X_gt.type() != "time_series" or X_syn.type() != "time_series":
            raise ValueError(
                f"Invalid data type gt = {X_gt.type()} syn = {X_syn.type()}"
            )

        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{X_gt.hash()}_{X_syn.hash()}_{self._reduction}_{platform.python_version()}_{platform.python_version()}.bkp"
        )
        if self.use_cache(cache_file):
            return load_from_file(cache_file)

        (
            id_static_gt,
            id_temporal_gt,
            id_observation_times_gt,
            id_outcome_gt,
        ) = X_gt.train().unpack(as_numpy=True)
        (
            ood_static_gt,
            ood_temporal_gt,
            ood_observation_times_gt,
            ood_outcome_gt,
        ) = X_gt.test().unpack(as_numpy=True)
        static_syn, temporal_syn, observation_times_syn, outcome_syn = X_syn.unpack(
            as_numpy=True
        )

        skf = KFold(
            n_splits=self._n_folds, shuffle=True, random_state=self._random_state
        )

        real_scores = []
        syn_scores_id = []
        syn_scores_ood = []

        def ts_eval_cbk(
            static_train: np.ndarray,
            temporal_train: np.ndarray,
            observation_times_train: np.ndarray,
            outcome_train: np.ndarray,
            static_test: np.ndarray,
            temporal_test: np.ndarray,
            observation_times_test: np.ndarray,
            outcome_test: np.ndarray,
        ) -> float:
            try:
                estimator = model(**model_args).fit(
                    static_train, temporal_train, observation_times_train, outcome_train
                )
                preds = estimator.predict(
                    static_test, temporal_test, observation_times_test
                )

                score = r2_score(outcome_test, preds)
            except BaseException as e:
                log.error(f"regression evaluation failed {e}")
                score = -1

            return score

        for train_idx, test_idx in skf.split(id_static_gt):
            static_train_data = id_static_gt[train_idx]
            temporal_train_data = id_temporal_gt[train_idx]
            observation_times_train_data = id_observation_times_gt[train_idx]
            outcome_train_data = id_outcome_gt[train_idx]

            static_test_data = id_static_gt[test_idx]
            temporal_test_data = id_temporal_gt[test_idx]
            observation_times_test_data = id_observation_times_gt[test_idx]
            outcome_test_data = id_outcome_gt[test_idx]

            real_score = ts_eval_cbk(
                static_train_data,
                temporal_train_data,
                observation_times_train_data,
                outcome_train_data,
                static_test_data,
                temporal_test_data,
                observation_times_test_data,
                outcome_test_data,
            )
            synth_score_id = ts_eval_cbk(
                static_syn,
                temporal_syn,
                observation_times_syn,
                outcome_syn,
                static_test_data,
                temporal_test_data,
                observation_times_test_data,
                outcome_test_data,
            )
            synth_score_ood = ts_eval_cbk(
                static_syn,
                temporal_syn,
                observation_times_syn,
                outcome_syn,
                ood_static_gt,
                ood_temporal_gt,
                ood_observation_times_gt,
                ood_outcome_gt,
            )

            real_scores.append(real_score)
            syn_scores_id.append(synth_score_id)
            syn_scores_ood.append(synth_score_ood)

        results = {
            "gt": float(self.reduction()(real_scores)),
            "syn_id": float(self.reduction()(syn_scores_id)),
            "syn_ood": float(self.reduction()(syn_scores_ood)),
        }

        save_to_file(cache_file, results)

        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate_time_series_survival_performance(
        self,
        model: Any,
        args: Dict,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        """Train a time series survival model on the synthetic data and evaluate the performance on real test data. Returns the average performance discrepancy between training on real data vs on synthetic data.

        Returns:
            gt and syn performance scores
        """
        if (
            X_gt.type() != "time_series_survival"
            or X_syn.type() != "time_series_survival"
        ):
            raise ValueError(
                f"Invalid data type gt = {X_gt.type()} syn = {X_syn.type()}"
            )

        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{X_gt.hash()}_{X_syn.hash()}_{self._reduction}_{platform.python_version()}_{platform.python_version()}.bkp"
        )
        if self.use_cache(cache_file):
            return load_from_file(cache_file)

        info = X_gt.info()
        time_horizons = info["time_horizons"]

        (
            id_X_static_gt,
            id_X_temporal_gt,
            id_X_observation_times_gt,
            id_T_gt,
            id_E_gt,
        ) = X_gt.train().unpack(as_numpy=True)
        (
            ood_X_static_gt,
            ood_X_temporal_gt,
            ood_X_observation_times_gt,
            ood_T_gt,
            ood_E_gt,
        ) = X_gt.test().unpack(as_numpy=True)
        (
            iter_X_static_syn,
            iter_X_temporal_syn,
            iter_X_observation_times_syn,
            iter_T_syn,
            iter_E_syn,
        ) = X_syn.unpack(as_numpy=True)

        predictor_gt = model(**args)
        log.info(
            f" Performance eval for df hash = {X_gt.train().hash()} ood hash = {X_gt.test().hash()}"
        )
        score_gt = evaluate_ts_survival_model(
            predictor_gt,
            id_X_static_gt,
            id_X_temporal_gt,
            id_X_observation_times_gt,
            id_T_gt,
            id_E_gt,
            metrics=["c_index", "brier_score"],
            n_folds=self._n_folds,
            time_horizons=time_horizons,
        )["clf"]

        log.info(f"Baseline performance score: {score_gt}")

        predictor_syn = model(**args)

        fail_score = {
            "c_index": (0, 0),
            "brier_score": (1, 0),
        }
        try:
            predictor_syn.fit(
                iter_X_static_syn,
                iter_X_temporal_syn,
                iter_X_observation_times_syn,
                iter_T_syn,
                iter_E_syn,
            )
            score_syn_id = evaluate_ts_survival_model(
                [predictor_syn] * self._n_folds,
                id_X_static_gt,
                id_X_temporal_gt,
                id_X_observation_times_gt,
                id_T_gt,
                id_E_gt,
                metrics=["c_index", "brier_score"],
                n_folds=self._n_folds,
                time_horizons=time_horizons,
                pretrained=True,
            )["clf"]
        except BaseException as e:
            log.error(
                f"Failed to evaluate synthetic ID performance. {model.name()}: {e}"
            )
            score_syn_id = fail_score

        log.info(f"Synthetic ID performance score: {score_syn_id}")

        try:
            predictor_syn.fit(
                iter_X_static_syn,
                iter_X_temporal_syn,
                iter_X_observation_times_syn,
                iter_T_syn,
                iter_E_syn,
            )
            score_syn_ood = evaluate_ts_survival_model(
                [predictor_syn] * self._n_folds,
                ood_X_static_gt,
                ood_X_temporal_gt,
                ood_X_observation_times_gt,
                ood_T_gt,
                ood_E_gt,
                metrics=["c_index", "brier_score"],
                n_folds=self._n_folds,
                time_horizons=time_horizons,
                pretrained=True,
            )["clf"]
        except BaseException as e:
            log.error(
                f"Failed to evaluate synthetic OOD performance. {model.name()}: {e}"
            )
            score_syn_ood = fail_score

        log.info(f"Synthetic OOD performance score: {score_syn_ood}")

        results = {
            "gt.c_index": float(score_gt["c_index"][0]),
            "gt.brier_score": float(score_gt["brier_score"][0]),
            "syn_id.c_index": float(score_syn_id["c_index"][0]),
            "syn_id.brier_score": float(score_syn_id["brier_score"][0]),
            "syn_ood.c_index": float(score_syn_ood["c_index"][0]),
            "syn_ood.brier_score": float(score_syn_ood["brier_score"][0]),
        }
        save_to_file(cache_file, results)

        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_default(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> float:
        results = self.evaluate(X_gt, X_syn)

        if self._task_type == "survival_analysis":
            return results["syn_id.c_index"] - results["syn_id.brier_score"]
        elif self._task_type == "classification" or self._task_type == "regression":
            return results["syn_id"]
        elif self._task_type == "time_series_survival":
            return results["syn_id.c_index"] - results["syn_id.brier_score"]
        else:
            raise RuntimeError(f"Unuspported task type {self._task_type}")


class PerformanceEvaluatorXGB(PerformanceEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_performance.PerformanceEvaluatorXGB
        :parts: 1


    Train an XGBoost classifier or regressor on the synthetic data and evaluate the performance on real test data.

    Returns the average performance discrepancy between training on real data vs on synthetic data.

    Score:
        close to 1: similar performance
        close to 0: massive performance degradation
    """

    @staticmethod
    def name() -> str:
        return "xgb"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        if self._task_type == "survival_analysis":
            return self._evaluate_survival_model(
                XGBSurvivalAnalysis,
                {
                    "n_jobs": 2,
                    "verbosity": 0,
                    "depth": 3,
                    "strategy": "weibull",  # "weibull", "debiased_bce"
                    "random_state": self._random_state,
                },
                X_gt,
                X_syn,
            )
        elif self._task_type == "classification" or self._task_type == "regression":
            xgb_clf_args = {
                "n_jobs": 2,
                "verbosity": 0,
                "depth": 3,
                "random_state": self._random_state,
            }

            xgb_reg_args = copy.deepcopy(xgb_clf_args)

            return self._evaluate_standard_performance(
                XGBClassifier,
                xgb_clf_args,
                XGBRegressor,
                xgb_reg_args,
                X_gt,
                X_syn,
            )
        elif self._task_type == "time_series_survival":
            return self._evaluate_time_series_survival_performance(
                XGBTimeSeriesSurvival,
                {
                    "n_jobs": 2,
                    "verbosity": 0,
                    "depth": 3,
                    "strategy": "weibull",  # "weibull", "debiased_bce"
                    "random_state": self._random_state,
                },
                X_gt,
                X_syn,
            )
        else:
            raise RuntimeError(f"Unuspported task type {self._task_type}")


class PerformanceEvaluatorLinear(PerformanceEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_performance.PerformanceEvaluatorLinear
        :parts: 1

    Train a Linear classifier or regressor on the synthetic data and evaluate the performance on real test data.

    Returns the average performance discrepancy between training on real data vs on synthetic data.

    Score:
        close to 1: similar performance
        close to 0: massive performance degradation
    """

    @staticmethod
    def name() -> str:
        return "linear_model"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        if self._task_type == "survival_analysis":
            return self._evaluate_survival_model(CoxPHSurvivalAnalysis, {}, X_gt, X_syn)
        elif self._task_type == "classification" or self._task_type == "regression":
            return self._evaluate_standard_performance(
                LogisticRegression,
                {"random_state": self._random_state},
                LinearRegression,
                {},
                X_gt,
                X_syn,
            )
        elif self._task_type == "time_series_survival":
            static, temporal, observation_times, T, E = X_gt.unpack()

            args: dict = {}
            log.info(f"Performance evaluation using CoxTimeSeriesSurvival and {args}")
            return self._evaluate_time_series_survival_performance(
                CoxTimeSeriesSurvival, args, X_gt, X_syn
            )
        else:
            raise RuntimeError(f"Unuspported task type {self._task_type}")


class PerformanceEvaluatorMLP(PerformanceEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_performance.PerformanceEvaluatorMLP
        :parts: 1

    Train a Neural Net classifier or regressor on the synthetic data and evaluate the performance on real test data.

    Returns the average performance discrepancy between training on real data vs on synthetic data.

    Score:
        close to 1: similar performance
        close to 1: massive performance degradation
    """

    @staticmethod
    def name() -> str:
        return "mlp"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        if self._task_type == "survival_analysis":
            return self._evaluate_survival_model(
                DeephitSurvivalAnalysis, {}, X_gt, X_syn
            )

        elif self._task_type == "classification" or self._task_type == "regression":
            mlp_args = {
                "n_units_in": X_gt.shape[1] - 1,
                "n_units_out": 1,
                "random_state": self._random_state,
            }
            clf_args = copy.deepcopy(mlp_args)
            clf_args["task_type"] = "classification"
            reg_args = copy.deepcopy(mlp_args)
            reg_args["task_type"] = "regression"

            return self._evaluate_standard_performance(
                MLP,
                clf_args,
                MLP,
                reg_args,
                X_gt,
                X_syn,
            )
        elif self._task_type == "time_series":
            info = X_gt.info()
            args = {
                "task_type": "regression",
                "n_static_units_in": len(info["static_features"]),
                "n_temporal_units_in": len(info["temporal_features"]),
                "n_temporal_window": info["window_len"],
                "output_shape": [info["outcome_len"]],
            }
            return self._evaluate_time_series_performance(
                TimeSeriesModel, args, X_gt, X_syn
            )
        elif self._task_type == "time_series_survival":
            static, temporal, observation_times, T, E = X_gt.unpack()

            info = X_gt.info()

            args = {}
            log.info(
                f"Performance evaluation using DynamicDeephitTimeSeriesSurvival and {args}"
            )
            return self._evaluate_time_series_survival_performance(
                DynamicDeephitTimeSeriesSurvival, args, X_gt, X_syn
            )

        else:
            raise RuntimeError(f"Unuspported task type {self._task_type}")


class FeatureImportanceRankDistance(MetricEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_performance.FeatureImportanceRankDistance
        :parts: 1

    Train an XGBoost classifier or regressor on the synthetic data and evaluate the feature importance.
    Train an XGBoost model on the real data and evaluate the feature importance.

    Returns the rank distance between the feature importance
    Returns the average performance discrepancy between training on real data vs on synthetic data.

    Score:
        close to 1: similar performance
        close to 0: unrelated
        close to -1: the ranks have different monotony.
    """

    def __init__(self, distance: str = "kendall", **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if distance not in ["kendall", "spearman"]:
            raise ValueError(f"Invalid feature distance {distance}")

        self._distance = distance

    @staticmethod
    def type() -> str:
        return "performance"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @staticmethod
    def name() -> str:
        return "feat_rank_distance"

    def distance(self, lhs: np.ndarray, rhs: np.ndarray) -> Tuple[float, float]:
        if self._distance == "spearman":
            return spearmanr(lhs, rhs, nan_policy="omit")
        elif self._distance == "kendall":
            return kendalltau(lhs, rhs, nan_policy="omit")
        else:
            raise RuntimeError(f"unknown distance {self.distance}")

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{X_gt.hash()}_{X_syn.hash()}_{platform.python_version()}.bkp"
        )
        if self.use_cache(cache_file):
            results = load_from_file(cache_file)
            log.info(
                f" Feature Importance rank distance df hash = {X_gt.train().hash()} ood hash = {X_gt.test().hash()}. score = {results}"
            )
            return results

        if self._task_type == "survival_analysis":
            model = XGBSurvivalAnalysis(
                n_jobs=2,
                verbosity=0,
                depth=3,
                strategy="weibull",  # "weibull", "debiased_bce"
                random_state=self._random_state,
            )

            id_X_gt, id_T_gt, id_E_gt = X_gt.train().unpack()
            ood_X_gt, ood_T_gt, ood_E_gt = X_gt.test().unpack()
            iter_X_syn, iter_T_syn, iter_E_syn = X_syn.unpack()

            columns = id_X_gt.columns

            gt_model = copy.deepcopy(model).fit(id_X_gt, id_T_gt, id_E_gt)
            gt_shap = gt_model.explain(ood_X_gt)

            syn_shap = np.random.rand(*ood_X_gt.shape)
            try:
                syn_model = copy.deepcopy(model).fit(iter_X_syn, iter_T_syn, iter_E_syn)
                syn_shap = syn_model.explain(ood_X_gt)
            except BaseException:
                pass

            syn_xai = np.mean(np.abs(syn_shap), axis=0)  # [n_features]
            gt_xai = np.mean(np.abs(gt_shap), axis=0)  # [n_features]
            if len(syn_xai) != len(columns):
                raise RuntimeError("Invalid xai features")

            corr, pvalue = self.distance(syn_xai, gt_xai)
            corr = np.mean(np.nan_to_num(corr))
            pvalue = np.mean(np.nan_to_num(pvalue))

            results = {
                "corr": corr,
                "pvalue": pvalue,
            }

        elif self._task_type == "classification":
            model = XGBClassifier(
                n_jobs=2,
                verbosity=0,
                depth=3,
                random_state=self._random_state,
            )

            id_X_gt, id_y_gt = X_gt.train().unpack()
            ood_X_gt, ood_y_gt = X_gt.test().unpack()
            iter_X_syn, iter_y_syn = X_syn.unpack()

            syn_shap = np.random.rand(
                len(np.unique(id_y_gt)), ood_X_gt.shape[0], ood_X_gt.shape[1]
            )
            try:
                syn_model = copy.deepcopy(model).fit(iter_X_syn, iter_y_syn)
                syn_explainer = shap.TreeExplainer(syn_model)
                syn_shap = syn_explainer.shap_values(ood_X_gt)
            except BaseException:
                pass

            gt_model = copy.deepcopy(model).fit(id_X_gt, id_y_gt)
            gt_explainer = shap.TreeExplainer(gt_model)
            gt_shap = gt_explainer.shap_values(ood_X_gt)

            # evaluate absolute influence for each class
            syn_xai = np.mean(np.abs(syn_shap), axis=1)  # classes x n_features
            gt_xai = np.mean(np.abs(gt_shap), axis=1)  # classes x n_features

            corr, pvalue = self.distance(syn_xai, gt_xai)
            corr = np.mean(np.nan_to_num(corr))
            pvalue = np.mean(np.nan_to_num(pvalue))

            results = {
                "corr": corr,
                "pvalue": pvalue,
            }

        elif self._task_type == "regression":
            model = XGBRegressor(
                n_jobs=2,
                verbosity=0,
                depth=3,
                random_state=self._random_state,
            )
            id_X_gt, id_y_gt = X_gt.train().unpack()
            ood_X_gt, ood_y_gt = X_gt.test().unpack()
            iter_X_syn, iter_y_syn = X_syn.unpack()

            syn_shap = np.random.rand(*ood_X_gt.shape)
            try:
                syn_model = copy.deepcopy(model).fit(iter_X_syn, iter_y_syn)
                syn_explainer = shap.TreeExplainer(syn_model)
                syn_shap = syn_explainer.shap_values(ood_X_gt)
            except BaseException:
                pass

            gt_model = copy.deepcopy(model).fit(id_X_gt, id_y_gt)
            gt_explainer = shap.TreeExplainer(gt_model)
            gt_shap = gt_explainer.shap_values(ood_X_gt)

            syn_xai = np.mean(np.abs(syn_shap), axis=0)  # [n_features]
            gt_xai = np.mean(np.abs(gt_shap), axis=0)  # [n_features]

            corr, pvalue = self.distance(syn_xai, gt_xai)
            corr = np.mean(np.nan_to_num(corr))
            pvalue = np.mean(np.nan_to_num(pvalue))

            results = {
                "corr": corr,
                "pvalue": pvalue,
            }
        else:
            raise RuntimeError(f"Unuspported task type {self._task_type}")

        save_to_file(cache_file, results)

        log.info(
            f" Feature Importance rank distance df hash = {X_gt.train().hash()} ood hash = {X_gt.test().hash()}. score = {results}"
        )
        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_default(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> float:
        results = self.evaluate(X_gt, X_syn)

        return results["corr"]
