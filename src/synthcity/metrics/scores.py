# stdlib
import multiprocessing
import time
from typing import Any, Tuple

# third party
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.stats import iqr

# synthcity absolute
import synthcity.logger as log

# synthcity relative
from .core.metric import MetricEvaluator

n_jobs = torch.cuda.device_count()
if n_jobs == 0:
    n_jobs = multiprocessing.cpu_count()

dispatcher = Parallel(n_jobs=n_jobs)


def _safe_evaluate(
    name: str,
    evaluator: MetricEvaluator,
    *args: Any,
    **kwargs: Any,
) -> Tuple[str, float, bool, float, str]:
    start = time.time()
    log.debug(f" >> Evaluating metric {name}")
    failed = False
    try:
        result = evaluator.evaluate(*args, **kwargs)
    except BaseException:
        result = 0
        failed = True

    duration = float(time.time() - start)
    log.debug(f" >> Evaluating metric {name} done. Duration: {duration} s")
    return name, result, failed, duration, evaluator.direction()


class ScoreEvaluator:
    def __init__(self) -> None:
        self.scores: dict = {}
        self.pending_tasks: list = []

    def add(
        self, key: str, result: float, failed: int, duration: float, direction: str
    ) -> None:
        if key not in self.scores:
            self.scores[key] = {
                "values": [],
                "errors": 0,
                "durations": [],
                "direction": direction,
            }
        self.scores[key]["values"].append(result)
        self.scores[key]["durations"].append(duration)
        self.scores[key]["errors"] += int(failed)

    def queue(
        self,
        key: str,
        evaluator: MetricEvaluator,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.pending_tasks.append((key, evaluator, args, kwargs))

    def compute(self) -> None:
        results = dispatcher(
            delayed(_safe_evaluate)(key, evaluator, *args, **kwargs)
            for (key, evaluator, args, kwargs) in self.pending_tasks
        )
        self.pending_tasks = []

        for key, result, failed, duration, direction in results:
            self.add(key, result, failed, duration, direction)

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
            "direction",
        ]
        output = pd.DataFrame([], columns=output_metrics)
        for metric in self.scores:
            values = self.scores[metric]["values"]
            errors = self.scores[metric]["errors"]
            direction = self.scores[metric]["direction"]
            durations = round(np.mean(self.scores[metric]["durations"]), 2)

            score_min = np.min(values)
            score_max = np.max(values)
            score_mean = np.mean(values)
            score_median = np.median(values)
            score_stddev = np.std(values)
            score_iqr = iqr(values)
            score_rounds = len(values)
            output = pd.concat(
                [
                    output,
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
                                direction,
                            ]
                        ],
                        columns=output_metrics,
                        index=[metric],
                    ),
                ],
            )

        return output
