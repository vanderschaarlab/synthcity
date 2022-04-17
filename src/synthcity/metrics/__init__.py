# stdlib
from typing import Dict, List, Optional

# third party
import pandas as pd
from pydantic import validate_arguments

# synthcity relative
from .eval_detection import (
    SyntheticDetectionGMM,
    SyntheticDetectionMLP,
    SyntheticDetectionXGB,
)
from .eval_performance import (
    PerformanceEvaluatorLinear,
    PerformanceEvaluatorMLP,
    PerformanceEvaluatorXGB,
)
from .eval_privacy import (
    DeltaPresence,
    IdentifiabilityScore,
    kAnonymization,
    kMap,
    lDiversityDistinct,
)
from .eval_sanity import (
    CloseValuesProbability,
    CommonRowsProportion,
    DataMismatchScore,
    DistantValuesProbability,
    NearestSyntheticNeighborDistance,
)
from .eval_statistical import (
    ChiSquaredTest,
    FeatureCorrelation,
    InverseKLDivergence,
    JensenShannonDistance,
    KolmogorovSmirnovTest,
    MaximumMeanDiscrepancy,
    WassersteinDistance,
)
from .scores import ScoreEvaluator

standard_metrics = [
    # sanity tests
    DataMismatchScore,
    CommonRowsProportion,
    NearestSyntheticNeighborDistance,
    CloseValuesProbability,
    DistantValuesProbability,
    # statistical tests
    JensenShannonDistance,
    ChiSquaredTest,
    FeatureCorrelation,
    InverseKLDivergence,
    KolmogorovSmirnovTest,
    MaximumMeanDiscrepancy,
    WassersteinDistance,
    # performance tests
    PerformanceEvaluatorLinear,
    PerformanceEvaluatorMLP,
    PerformanceEvaluatorXGB,
    # synthetic detection tests
    SyntheticDetectionXGB,
    SyntheticDetectionMLP,
    SyntheticDetectionGMM,
    # privacy tests
    DeltaPresence,
    kAnonymization,
    kMap,
    lDiversityDistinct,
    IdentifiabilityScore,
]


class Metrics:
    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        X_gt: pd.DataFrame,
        X_syn: pd.DataFrame,
        sensitive_columns: List[str] = [],
        reduction: str = "mean",
        n_histogram_bins: int = 10,
        metrics: Optional[Dict] = None,
    ) -> pd.DataFrame:
        if metrics is None:
            metrics = Metrics.list()

        scores = ScoreEvaluator()

        for metric in standard_metrics:
            if metric.type() not in metrics:
                continue
            if metric.name() not in metrics[metric.type()]:
                continue
            scores.queue(
                metric(
                    sensitive_columns=sensitive_columns,
                    reduction=reduction,
                    n_histogram_bins=n_histogram_bins,
                ),
                X_gt,
                X_syn,
            )

        scores.compute()

        return scores.to_dataframe()

    @staticmethod
    def list() -> dict:
        available_metrics: Dict[str, List] = {}
        for metric in standard_metrics:
            if metric.type() not in available_metrics:
                available_metrics[metric.type()] = []
            available_metrics[metric.type()].append(metric.name())

        return available_metrics
