# stdlib
from typing import Dict, List, Optional

# third party
import pandas as pd
from pydantic import validate_arguments

# synthcity relative
from .attacks import DataLeakageLinear, DataLeakageMLP, DataLeakageXGB
from .detection import (
    SyntheticDetectionGMM,
    SyntheticDetectionMLP,
    SyntheticDetectionXGB,
)
from .performance import (
    PerformanceEvaluatorLinear,
    PerformanceEvaluatorMLP,
    PerformanceEvaluatorXGB,
)
from .privacy import DeltaPresence, kAnonymization, kMap, lDiversity
from .sanity import (
    CommonRowsProportion,
    DataMismatchScore,
    InlierProbability,
    NearestSyntheticNeighborDistance,
    OutlierProbability,
)
from .scores import ScoreEvaluator
from .statistical import (
    ChiSquaredTest,
    FeatureCorrelation,
    InverseCDFDistance,
    InverseKLDivergence,
    JensenShannonDistance,
    KolmogorovSmirnovTest,
    MaximumMeanDiscrepancy,
    WassersteinDistance,
)

standard_metrics = [
    # sanity tests
    DataMismatchScore,
    CommonRowsProportion,
    NearestSyntheticNeighborDistance,
    InlierProbability,
    OutlierProbability,
    # statistical tests
    JensenShannonDistance,
    ChiSquaredTest,
    FeatureCorrelation,
    InverseCDFDistance,
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
    # attacks
    DataLeakageMLP,
    DataLeakageLinear,
    DataLeakageXGB,
    # privacy tests
    DeltaPresence,
    kAnonymization,
    kMap,
    lDiversity,
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
            print(metric)
            if metric.type() not in metrics:
                continue
            if metric.name() not in metrics[metric.type()]:
                continue
            key = f"{metric.type()}.{metric.name()}"
            scores.queue(
                key,
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
