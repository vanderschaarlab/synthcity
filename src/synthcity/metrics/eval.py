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
    AugmentationPerformanceEvaluatorLinear,
    AugmentationPerformanceEvaluatorMLP,
    AugmentationPerformanceEvaluatorXGB,
    FeatureImportanceRankDistance,
    PerformanceEvaluatorLinear,
    PerformanceEvaluatorMLP,
    PerformanceEvaluatorXGB,
)
from .eval_privacy import (
    DeltaPresence,
    DomiasMIABNAF,
    DomiasMIAKDE,
    DomiasMIAPrior,
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
    FrechetInceptionDistance,
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
    InverseKLDivergence,
    KolmogorovSmirnovTest,
    MaximumMeanDiscrepancy,
    WassersteinDistance,
    PRDCScore,
    AlphaPrecision,
    SurvivalKMDistance,
    FrechetInceptionDistance,
    # performance tests
    PerformanceEvaluatorLinear,
    PerformanceEvaluatorMLP,
    PerformanceEvaluatorXGB,
    AugmentationPerformanceEvaluatorLinear,
    AugmentationPerformanceEvaluatorMLP,
    AugmentationPerformanceEvaluatorXGB,
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
    DomiasMIABNAF,
    DomiasMIAKDE,
    DomiasMIAPrior,
]


class Metrics:
    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        X_gt: Union[DataLoader, pd.DataFrame],
        X_syn: Union[DataLoader, pd.DataFrame],
        X_train: Optional[Union[DataLoader, pd.DataFrame]] = None,
        X_ref_syn: Optional[Union[DataLoader, pd.DataFrame]] = None,
        X_augmented: Optional[Union[DataLoader, pd.DataFrame]] = None,
        reduction: str = "mean",
        n_histogram_bins: int = 10,
        metrics: Optional[Dict] = None,
        task_type: str = "classification",
        random_state: int = 0,
        workspace: Path = Path("workspace"),
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Core evaluation logic for the metrics

        X_gt: Dataloader or DataFrame
            Reference real data
        X_syn: Dataloader or DataFrame
            Synthetic data
        X_train: Dataloader or DataFrame
            The data used to train the synthetic model (used for domias metrics only).
        X_ref_syn: Dataloader or DataFrame
            Reference synthetic data (used for domias metrics only).
        X_augmented: Dataloader or DataFrame
            Augmented data
        metrics: dict
            the dictionary of metrics to evaluate
            Full dictionary of metrics is:
            {
                'sanity': ['data_mismatch', 'common_rows_proportion', 'nearest_syn_neighbor_distance', 'close_values_probability', 'distant_values_probability'],
                'stats': ['jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 'prdc', 'alpha_precision', 'survival_km_distance'],
                'performance': ['linear_model', 'mlp', 'xgb', 'feat_rank_distance'],
                'detection': ['detection_xgb', 'detection_mlp', 'detection_gmm', 'detection_linear'],
                'privacy': ['delta-presence', 'k-anonymization', 'k-map', 'distinct l-diversity', 'identifiability_score']
            }
        reduction: str
            The way to aggregate metrics across folds. Can be: 'mean', "min", or "max".
        n_histogram_bins: int
            The number of bins used in histogram calculation of a given metric. Defaults to 10.
        task_type: str
            The type of problem. Relevant for evaluating the downstream models with the correct metrics. Valid tasks are:  "classification", "regression", "survival_analysis", "time_series", "time_series_survival".
        random_state: int
            random seed
        workspace: Path
            The folder for caching intermediary results.
        use_cache: bool
            If the a metric has been previously run and is cached, it will be reused for the experiments. Defaults to True.
        """
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
        if X_train is not None and not isinstance(X_train, DataLoader):
            X_train = GenericDataLoader(X_train)
        if X_ref_syn is not None and not isinstance(X_ref_syn, DataLoader):
            X_ref_syn = create_from_info(X_ref_syn, X_gt.info())

        if X_gt.type() != X_syn.type():
            raise ValueError("Different dataloader types")

        if task_type == "survival_analysis":
            if (
                X_gt.type() != "survival_analysis"
                and X_train.type() != "survival_analysis"
            ):
                raise ValueError("Invalid dataloader for survival analysis")
        elif task_type == "time_series":
            if X_gt.type() != "time_series" and X_train.type() != "time_series":
                raise ValueError("Invalid dataloader for time series")
        elif task_type == "time_series_survival":
            if (
                X_gt.type() != "time_series_survival"
                and X_train.type() != "time_series_survival"
            ):
                raise ValueError("Invalid dataloader for time series survival analysis")

        if metrics is None:
            metrics = Metrics.list()

        X_gt, _ = X_gt.encode()
        X_syn, _ = X_syn.encode()

        if X_train:
            X_train, _ = X_train.encode()
        if X_ref_syn:
            X_ref_syn, _ = X_ref_syn.encode()
        if X_augmented:
            X_augmented, _ = X_augmented.encode()

        scores = ScoreEvaluator()

        eval_cnt = min(len(X_gt), len(X_syn))
        for metric in standard_metrics:
            if metric.type() not in metrics:
                continue
            if metric.name() not in metrics[metric.type()]:
                continue
            if X_augmented and "augmentation" in metric.name():
                scores.queue(
                    metric(
                        reduction=reduction,
                        n_histogram_bins=n_histogram_bins,
                        task_type=task_type,
                        random_state=random_state,
                        workspace=workspace,
                        use_cache=use_cache,
                    ),
                    X_gt,
                    X_augmented,
                )
            elif "DomiasMIA" in metric.name():
                scores.queue(
                    metric(
                        reduction=reduction,
                        n_histogram_bins=n_histogram_bins,
                        task_type=task_type,
                        random_state=random_state,
                        workspace=workspace,
                        use_cache=use_cache,
                    ),
                    X_gt,
                    X_syn,
                    X_train,
                    X_ref_syn,
                    reference_size=10,  # TODO: review this
                )
            else:
                scores.queue(
                    metric(
                        reduction=reduction,
                        n_histogram_bins=n_histogram_bins,
                        task_type=task_type,
                        random_state=random_state,
                        workspace=workspace,
                        use_cache=use_cache,
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
