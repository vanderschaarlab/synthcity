# stdlib
from pathlib import Path
from typing import Dict, List, Optional, Union

# third party
import pandas as pd
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.dataloader import (
    DataLoader,
    GenericDataLoader,
    create_from_info,
)

# synthcity relative
from .eval_detection import (
    SyntheticDetectionGMM,
    SyntheticDetectionLinear,
    SyntheticDetectionMLP,
    SyntheticDetectionXGB,
)
from .eval_performance import (
    FeatureImportanceRankDistance,
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
    AlphaPrecision,
    ChiSquaredTest,
    FeatureCorrelation,
    InverseKLDivergence,
    JensenShannonDistance,
    KolmogorovSmirnovTest,
    MaximumMeanDiscrepancy,
    PRDCScore,
    SurvivalKMDistance,
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
    PRDCScore,
    AlphaPrecision,
    SurvivalKMDistance,
    # performance tests
    PerformanceEvaluatorLinear,
    PerformanceEvaluatorMLP,
    PerformanceEvaluatorXGB,
    FeatureImportanceRankDistance,
    # synthetic detection tests
    SyntheticDetectionXGB,
    SyntheticDetectionMLP,
    SyntheticDetectionGMM,
    SyntheticDetectionLinear,
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
        X_gt: Union[DataLoader, pd.DataFrame],
        X_syn: Union[DataLoader, pd.DataFrame],
        reduction: str = "mean",
        n_histogram_bins: int = 10,
        metrics: Optional[Dict] = None,
        task_type: str = "classification",
        random_state: int = 0,
        workspace: Path = Path("workspace"),
    ) -> pd.DataFrame:
        workspace.mkdir(parents=True, exist_ok=True)

        supported_tasks = [
            "classification",
            "regression",
            "survival_analysis",
            "time_series",
            "time_series_survival",
        ]
        if task_type not in supported_tasks:
            raise ValueError(
                f"Invalid task type {task_type}. Supported: {supported_tasks}"
            )

        if not isinstance(X_gt, DataLoader):
            X_gt = GenericDataLoader(X_gt)
        if not isinstance(X_syn, DataLoader):
            X_syn = create_from_info(X_syn, X_gt.info())

        if X_gt.type() != X_syn.type():
            raise ValueError("Different dataloader types")

        if task_type == "survival_analysis":
            if X_gt.type() != "survival_analysis":
                raise ValueError("Invalid dataloader for survival analysis")
        elif task_type == "time_series":
            if X_gt.type() != "time_series":
                raise ValueError("Invalid dataloader for time series")
        elif task_type == "time_series_survival":
            if X_gt.type() != "time_series_survival":
                raise ValueError("Invalid dataloader for time series survival analysis")

        if metrics is None:
            metrics = Metrics.list()

        X_gt, _ = X_gt.encode()
        X_syn, _ = X_syn.encode()

        scores = ScoreEvaluator()

        eval_cnt = min(len(X_gt), len(X_syn))

        for metric in standard_metrics:
            if metric.type() not in metrics:
                continue
            if metric.name() not in metrics[metric.type()]:
                continue
            scores.queue(
                metric(
                    reduction=reduction,
                    n_histogram_bins=n_histogram_bins,
                    task_type=task_type,
                    random_state=random_state,
                    workspace=workspace,
                ),
                X_gt.sample(eval_cnt),
                X_syn.sample(eval_cnt),
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
