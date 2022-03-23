# stdlib
import multiprocessing
import time
from typing import Any, Callable, Tuple

# third party
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import iqr

dispatcher = Parallel(n_jobs=multiprocessing.cpu_count())


def _safe_evaluate(
    name: str, cbk: Callable, *args: Any, **kwargs: Any
) -> Tuple[str, float, bool, float]:
    start = time.time()
    failed = False
    try:
        result = cbk(*args, **kwargs)
    except BaseException:
        result = 0
        failed = True

    duration = float(time.time() - start)
    return name, result, failed, duration


class ScoreEvaluator:
    def __init__(self, repeats: int) -> None:
        self.scores: dict = {}
        self.repeats = repeats
        self.pending_tasks: list = []

    def queue(self, key: str, cbk: Callable, *args: Any, **kwargs: Any) -> None:
        for repeat in range(self.repeats):
            self.pending_tasks.append((key, cbk, args, kwargs))

    def compute(self) -> None:
        results = dispatcher(
            delayed(_safe_evaluate)(key, cbk, *args, **kwargs)
            for (key, cbk, args, kwargs) in self.pending_tasks
        )
        self.pending_tasks = []

        for key, result, failed, duration in results:
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
