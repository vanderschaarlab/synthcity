# stdlib
from typing import Dict, List, Optional

# third party
import pandas as pd
from pydantic import validate_arguments

# synthcity absolute
from synthcity.utils.scores import ScoreEvaluator

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
    "privacy": {
        "k_anonymizatio": evaluate_k_anonymization,
        "l_diversity": evaluate_l_diversity,
    }
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


class Metrics:
    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        X_gt: pd.DataFrame,
        y_gt: pd.Series,
        X_syn: pd.DataFrame,
        y_syn: pd.Series,
        sensitive_columns: List[str] = [],
        metrics: Optional[Dict] = None,
        repeats: int = 3,
    ) -> pd.DataFrame:
        if metrics is None:
            metrics = Metrics.list()

        scores = ScoreEvaluator()

        for repeat in range(repeats):
            for category in standard_metrics:
                if category not in metrics:
                    continue
                for metric in standard_metrics[category]:
                    if metric not in metrics[category]:
                        continue
                    key = f"{category}.{metric}"
                    scores.queue(
                        key,
                        standard_metrics[category][metric],
                        X_gt,
                        y_gt,
                        X_syn,
                        y_syn,
                    )

            for category in unary_privacy_metrics:
                if category not in metrics:
                    continue
                for metric in unary_privacy_metrics[category]:
                    if metric not in metrics[category]:
                        continue
                    for name, src in [("gt", X_gt), ("syn", X_syn)]:
                        key = f"{category}.{metric}.{name}"
                        scores.queue(
                            key,
                            unary_privacy_metrics[category][metric],
                            src,
                            sensitive_columns,
                        )

            for category in binary_privacy_metrics:
                if category not in metrics:
                    continue
                for metric in binary_privacy_metrics[category]:
                    if metric not in metrics[category]:
                        continue
                    key = f"{category}.{metric}"
                    scores.queue(
                        key,
                        binary_privacy_metrics[category][metric],
                        X_gt,
                        X_syn,
                        sensitive_columns,
                    )

        scores.compute()

        return scores.to_dataframe()

    @staticmethod
    def list() -> dict:
        available_metrics = {}
        for category in standard_metrics:
            available_metrics[category] = list(standard_metrics[category].keys())

        for category in unary_privacy_metrics:
            if category not in available_metrics:
                available_metrics[category] = []
            available_metrics[category].extend(
                list(unary_privacy_metrics[category].keys())
            )

        for category in binary_privacy_metrics:
            if category not in available_metrics:
                available_metrics[category] = []
            available_metrics[category].extend(
                list(binary_privacy_metrics[category].keys())
            )

        return available_metrics
