# stdlib
from typing import Dict, List

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from scipy.stats import iqr

# synthcity relative
from .basic import (
    evaluate_avg_distance_nearest_synth_neighbor,
    evaluate_common_rows_proportion,
    evaluate_data_mismatch_score,
    evaluate_inlier_probability,
    evaluate_outlier_probability,
)
from .detection import (
    evaluate_gmm_detection_synthetic,
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
    },
    "performance": {
        "train_synth_test_real_data": evaluate_test_performance,
    },
    "detection": {
        "gmm_detection": evaluate_gmm_detection_synthetic,
        "xgb_detection": evaluate_xgb_detection_synthetic,
    },
}
unary_privacy_metrics = {
    "k_anonymizatio": evaluate_k_anonymization,
    "l_diversity": evaluate_l_diversity,
}

binary_privacy_metrics = {
    "kmap": evaluate_kmap,
    "delta_presence": evaluate_delta_presence,
}


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate(
    X_gt: pd.DataFrame,
    y_gt: pd.Series,
    X_syn: pd.DataFrame,
    y_syn: pd.Series,
    sensitive_columns: List[str] = [],
    repeats: int = 10,
) -> Dict:
    scores: dict = {}
    for retry in range(repeats):
        for category in standard_metrics:
            for metric in standard_metrics[category]:
                key = f"{category}.{metric}"
                if key not in scores:
                    scores[key] = []
                scores[key].append(
                    standard_metrics[category][metric](X_gt, y_gt, X_syn, y_syn)
                )
        for metric in unary_privacy_metrics:
            for name, src in [("gt", X_gt), ("syn", X_syn)]:
                score = unary_privacy_metrics[metric](src, sensitive_columns)
                key = f"privacy.{metric}.{name}"

                if key not in scores:
                    scores[key] = []
                scores[key].append(score)
        for metric in binary_privacy_metrics:
            score = binary_privacy_metrics[metric](X_gt, X_syn, sensitive_columns)
            key = f"privacy.{metric}"

            if key not in scores:
                scores[key] = []
            scores[key].append(score)

    output_metrics = ["min", "max", "mean", "stddev", "median", "iqr", "rounds"]
    output = pd.DataFrame([], columns=output_metrics)
    for metric in scores:
        score_min = np.min(scores[metric])
        score_max = np.max(scores[metric])
        score_mean = np.mean(scores[metric])
        score_median = np.median(scores[metric])
        score_stddev = np.std(scores[metric])
        score_iqr = iqr(scores[metric])
        score_rounds = repeats
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
                    ]
                ],
                columns=output_metrics,
                index=[metric],
            )
        )
    return output
