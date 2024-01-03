# stdlib
from pathlib import Path
from typing import Any, List, Tuple, Union

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.dataloader import (
    DataLoader,
    GenericDataLoader,
    create_from_info,
)

# synthcity relative
from .eval import standard_metrics


class WeightedMetrics:
    def __init__(
        self,
        metrics: List[Tuple[str, str]],  # (type, name)
        weights: List[float],
        task_type: str = "classification",
        random_state: int = 0,
        workspace: Path = Path("workspace"),
    ) -> None:
        if len(metrics) != len(weights):
            raise ValueError("Metrics and weights should have the same length")

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

        self.weights = weights / (np.sum(weights) + 1e-8)
        self.workspace = workspace
        self.task_type = task_type
        self.random_state = random_state
        self.metrics = []

        directions = []
        for mtype, mname in metrics:
            runner: Any = None
            for ref_metric in standard_metrics:
                if ref_metric.type() != mtype:
                    continue
                if ref_metric.name() != mname:
                    continue
                runner = ref_metric(
                    task_type=task_type,
                    random_state=random_state,
                    workspace=workspace,
                )
            if runner is None:
                raise ValueError(f"Unknown metric {mtype} - {mname}")
            self.metrics.append(runner)
            directions.append(runner.direction())

        if len(np.unique(directions)) != 1:
            raise ValueError("Metrics have different evaluation directions.")

        self._direction = directions[0]

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: Union[DataLoader, pd.DataFrame],
        X_syn: Union[DataLoader, pd.DataFrame],
    ) -> float:
        if not isinstance(X_gt, DataLoader):
            X_gt = GenericDataLoader(X_gt)
        if not isinstance(X_syn, DataLoader):
            X_syn = create_from_info(X_syn, X_gt.info())

        if X_gt.type() != X_syn.type():
            raise ValueError("Different dataloader types")

        if self.task_type == "survival_analysis":
            if X_gt.type() != "survival_analysis":
                raise ValueError("Invalid dataloader for survival analysis")
        elif self.task_type == "time_series":
            if X_gt.type() != "time_series":
                raise ValueError("Invalid dataloader for time series")
        elif self.task_type == "time_series_survival":
            if X_gt.type() != "time_series_survival":
                raise ValueError("Invalid dataloader for time series survival analysis")

        score = 0
        eval_cnt = min(len(X_gt), len(X_syn))

        for weight, metric in zip(self.weights, self.metrics):
            score += weight * metric.evaluate_default(
                X_gt.sample(eval_cnt), X_syn.sample(eval_cnt)
            )

        return score

    def direction(self) -> str:
        return self._direction
