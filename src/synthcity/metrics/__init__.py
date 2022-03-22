# stdlib
import time
from typing import Any, Callable, List, Tuple

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from scipy.stats import iqr

# synthcity relative
from .attacks import (
    evaluate_sensitive_data_leakage_linear,
    evaluate_sensitive_data_leakage_mlp,
    evaluate_sensitive_data_leakage_xgb,
)
from .basic import (
    evaluate_avg_distance_nearest_synth_neighbor,
    evaluate_common_rows_proportion,
    evaluate_data_mismatch_score,
    evaluate_inlier_probability,
    evaluate_outlier_probability,
)
from .detection import (
    evaluate_gmm_detection_synthetic,
    evaluate_mlp_detection_synthetic,
    evaluate_xgb_detection_synthetic,
)
from .performance import evaluate_test_performance
from .privacy import (
    evaluate_delta_presence,
    evaluate_k_anonymization,
    evaluate_kmap,
    evaluate_l_diversity,
)
from .statistical import (
    evaluate_chi_squared_test,
    evaluate_inv_cdf_distance,
    evaluate_inv_kl_divergence,
    evaluate_kolmogorov_smirnov_test,
    evaluate_maximum_mean_discrepancy,
)

standard_metrics = {
    "sanity": {
        "data_mismatch_score": evaluate_data_mismatch_score,
        "common_rows_proportion": evaluate_common_rows_proportion,
        "avg_distance_nearest_neighbor": evaluate_avg_distance_nearest_synth_neighbor,
        "inlier_probability": evaluate_inlier_probability,
        "outlier_probability": evaluate_outlier_probability,
    },
    "statistical": {
        "inverse_kl_divergence": evaluate_inv_kl_divergence,
        "kolmogorov_smirnov_test": evaluate_kolmogorov_smirnov_test,
        "chi_squared_test": evaluate_chi_squared_test,
        "maximum_mean_discrepancy": evaluate_maximum_mean_discrepancy,
        "inverse_cdf_distance": evaluate_inv_cdf_distance,
    },
    "performance": {
        "train_synth_test_real_data": evaluate_test_performance,
    },
    "detection": {
        "gmm_detection": evaluate_gmm_detection_synthetic,
        "xgb_detection": evaluate_xgb_detection_synthetic,
        "mlp_detection": evaluate_mlp_detection_synthetic,
    },
}
unary_privacy_metrics = {
    "k_anonymizatio": evaluate_k_anonymization,
    "l_diversity": evaluate_l_diversity,
}

binary_privacy_metrics = {
    "privacy": {
        "kmap": evaluate_kmap,
        "delta_presence": evaluate_delta_presence,
    },
    "attacks": {
        "sensitive_data_reidentification_xgb": evaluate_sensitive_data_leakage_xgb,
        "sensitive_data_reidentification_mlp": evaluate_sensitive_data_leakage_mlp,
        "sensitive_data_reidentification_linear": evaluate_sensitive_data_leakage_linear,
    },
}


class MetricEvaluator:
    def __init__(self, repeats: int = 2) -> None:
        self.scores: dict = {}
        self.repeats = repeats

    def _safe_evaluate(
        self, cbk: Callable, *args: Any, **kwargs: Any
    ) -> Tuple[float, bool]:
        try:
            return cbk(*args, **kwargs), False
        except BaseException:
            return 0, True

    def score(self, key: str, cbk: Callable, *args: Any, **kwargs: Any) -> None:
        for repeat in range(self.repeats):
            start = time.time()
            result, failed = self._safe_evaluate(cbk, *args, **kwargs)
            duration = float(time.time() - start)

            if key not in self.scores:
                self.scores[key] = {
                    "values": [],
                    "errors": 0,
                    "durations": [],
                }
            self.scores[key]["values"].append(result)
            self.scores[key]["durations"].append(duration)
            self.scores[key]["errors"] += int(failed)

    def to_dataframe(self) -> pd.DataFrame:
        output_metrics = [
            "min",
            "max",
            "mean",
            "stddev",
            "median",
            "iqr",
            "rounds",
            "errors",
            "durations",
        ]
        output = pd.DataFrame([], columns=output_metrics)
        for metric in self.scores:
            values = self.scores[metric]["values"]
            errors = self.scores[metric]["errors"]
            durations = round(np.mean(self.scores[metric]["durations"]), 2)

            score_min = np.min(values)
            score_max = np.max(values)
            score_mean = np.mean(values)
            score_median = np.median(values)
            score_stddev = np.std(values)
            score_iqr = iqr(values)
            score_rounds = self.repeats
            output = output.append(
                pd.DataFrame(
                    [
                        [
                            score_min,
                            score_max,
                            score_mean,
                            score_stddev,
                            score_median,
                            score_iqr,
                            score_rounds,
                            errors,
                            durations,
                        ]
                    ],
                    columns=output_metrics,
                    index=[metric],
                )
            )

        return output


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate(
    X_gt: pd.DataFrame,
    y_gt: pd.Series,
    X_syn: pd.DataFrame,
    y_syn: pd.Series,
    sensitive_columns: List[str] = [],
    repeats: int = 2,
) -> pd.DataFrame:
    scores = MetricEvaluator(repeats=repeats)

    for category in standard_metrics:
        for metric in standard_metrics[category]:
            key = f"{category}.{metric}"
            scores.score(
                key, standard_metrics[category][metric], X_gt, y_gt, X_syn, y_syn
            )

    for metric in unary_privacy_metrics:
        for name, src in [("gt", X_gt), ("syn", X_syn)]:
            key = f"privacy.{metric}.{name}"
            scores.score(key, unary_privacy_metrics[metric], src, sensitive_columns)

    for category in binary_privacy_metrics:
        for metric in binary_privacy_metrics[category]:
            key = f"{category}.{metric}"
            scores.score(
                key,
                binary_privacy_metrics[category][metric],
                X_gt,
                X_syn,
                sensitive_columns,
            )

    return scores.to_dataframe()
