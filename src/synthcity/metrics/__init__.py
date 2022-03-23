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
    evaluate_avg_jensenshannon_distance,
    evaluate_chi_squared_test,
    evaluate_feature_correlation,
    evaluate_inv_cdf_distance,
    evaluate_inv_kl_divergence,
    evaluate_kolmogorov_smirnov_test,
    evaluate_maximum_mean_discrepancy,
)

standard_metrics = {
    "sanity": {
        "data_mismatch_score": {
            "cbk": evaluate_data_mismatch_score,
            "bad_score": 1,
            "ok_score": 0,
        },
        "common_rows_proportion": {
            "cbk": evaluate_common_rows_proportion,
            "bad_score": 1,
            "ok_score": 0,
        },
        "avg_distance_nearest_neighbor": {
            "cbk": evaluate_avg_distance_nearest_synth_neighbor,
            "bad_score": 0,
            "ok_score": 1,
        },
        "inlier_probability": {
            "cbk": evaluate_inlier_probability,
            "bad_score": 0,
            "ok_score": 1,
        },
        "outlier_probability": {
            "cbk": evaluate_outlier_probability,
            "bad_score": 1,
            "ok_score": 0,
        },
    },
    "statistical": {
        "inverse_kl_divergence": {
            "cbk": evaluate_inv_kl_divergence,
            "ok_score": 1,
            "bad_score": 0,
        },
        "kolmogorov_smirnov_test": {
            "cbk": evaluate_kolmogorov_smirnov_test,
            "ok_score": 1,
            "bad_score": 0,
        },
        "chi_squared_test": {
            "cbk": evaluate_chi_squared_test,
            "ok_score": 1,
            "bad_score": 0,
        },
        "maximum_mean_discrepancy": {
            "cbk": evaluate_maximum_mean_discrepancy,
            "ok_score": 0,
            "bad_score": 1,
        },
        "inverse_cdf_distance": {
            "cbk": evaluate_inv_cdf_distance,
            "ok_score": 0,
            "bad_score": 1,
        },
        "avg_jensenshannon_distance": {
            "cbk": evaluate_avg_jensenshannon_distance,
            "ok_score": 0,
            "bad_score": 1,
        },
        "feature_correlation": {
            "cbk": evaluate_feature_correlation,
            "ok_score": 0,
            "bad_score": 1,
        },
    },
    "performance": {
        "train_synth_test_real_data": {
            "cbk": evaluate_test_performance,
            "ok_score": 0,
            "bad_score": 1,
        },
    },
    "detection": {
        "gmm_detection": {
            "cbk": evaluate_gmm_detection_synthetic,
            "ok_score": 0,
            "bad_score": 1,
        },
        "xgb_detection": {
            "cbk": evaluate_xgb_detection_synthetic,
            "ok_score": 0,
            "bad_score": 1,
        },
        "mlp_detection": {
            "cbk": evaluate_mlp_detection_synthetic,
            "ok_score": 0,
            "bad_score": 1,
        },
    },
}
unary_privacy_metrics = {
    "privacy": {
        "k_anonymization": {
            "cbk": evaluate_k_anonymization,
            "ok_score": 10,
            "bad_score": 1,
        },
        "l_diversity": {
            "cbk": evaluate_l_diversity,
            "ok_score": 10,
            "bad_score": 1,
        },
    }
}

binary_privacy_metrics = {
    "privacy": {
        "kmap": {
            "cbk": evaluate_kmap,
            "ok_score": 10,
            "bad_score": 1,
        },
        "delta_presence": {
            "cbk": evaluate_delta_presence,
            "ok_score": 0,
            "bad_score": 1,
        },
    },
    "attacks": {
        "sensitive_data_reidentification_xgb": {
            "cbk": evaluate_sensitive_data_leakage_xgb,
            "ok_score": 0,
            "bad_score": 1,
        },
        "sensitive_data_reidentification_mlp": {
            "cbk": evaluate_sensitive_data_leakage_mlp,
            "ok_score": 0,
            "bad_score": 1,
        },
        "sensitive_data_reidentification_linear": {
            "cbk": evaluate_sensitive_data_leakage_linear,
            "ok_score": 0,
            "bad_score": 1,
        },
    },
}


class Metrics:
    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        X_gt: pd.DataFrame,
        X_syn: pd.DataFrame,
        sensitive_columns: List[str] = [],
        metrics: Optional[Dict] = None,
    ) -> pd.DataFrame:
        if metrics is None:
            metrics = Metrics.list()

        scores = ScoreEvaluator()

        for category in standard_metrics:
            if category not in metrics:
                continue
            for metric in standard_metrics[category]:
                if metric not in metrics[category]:
                    continue
                key = f"{category}.{metric}"
                scores.queue(
                    key,
                    standard_metrics[category][metric]["cbk"],
                    standard_metrics[category][metric]["ok_score"],
                    standard_metrics[category][metric]["bad_score"],
                    X_gt,
                    X_syn,
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
                        unary_privacy_metrics[category][metric]["cbk"],
                        unary_privacy_metrics[category][metric]["ok_score"],
                        unary_privacy_metrics[category][metric]["bad_score"],
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
                    binary_privacy_metrics[category][metric]["cbk"],
                    binary_privacy_metrics[category][metric]["ok_score"],
                    binary_privacy_metrics[category][metric]["bad_score"],
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
