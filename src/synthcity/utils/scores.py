# stdlib
import multiprocessing
import time
from typing import Any, Callable, Tuple

# third party
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import iqr

# synthcity absolute
import synthcity.logger as log

dispatcher = Parallel(n_jobs=multiprocessing.cpu_count())


def _safe_evaluate(
    name: str,
    cbk: Callable,
    ok_score: float,
    bad_score: float,
    *args: Any,
    **kwargs: Any,
) -> Tuple[str, float, bool, float, float, float]:
    start = time.time()
    log.info(f" >> Evaluating metric {name}")
    failed = False
    try:
        result = cbk(*args, **kwargs)
    except BaseException:
        result = 0
        failed = True

    duration = float(time.time() - start)
    log.debug(f" >> Evaluating metric {name} done. Duration: {duration} s")
    return name, result, failed, duration, ok_score, bad_score


class ScoreEvaluator:
    def __init__(self) -> None:
        self.scores: dict = {}
        self.pending_tasks: list = []

    def add(
        self,
        key: str,
        result: float,
        failed: int,
        duration: float,
        ok_score: float,
        bad_score: float,
    ) -> None:
        if key not in self.scores:
            self.scores[key] = {
                "values": [],
                "errors": 0,
                "durations": [],
                "ok_score": ok_score,
                "bad_score": bad_score,
            }
        self.scores[key]["values"].append(result)
        self.scores[key]["durations"].append(duration)
        self.scores[key]["errors"] += int(failed)

    def queue(
        self,
        key: str,
        cbk: Callable,
        ok_score: float,
        bad_score: float,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.pending_tasks.append((key, cbk, ok_score, bad_score, args, kwargs))

    def compute(self) -> None:
        results = dispatcher(
            delayed(_safe_evaluate)(key, cbk, ok_score, bad_score, *args, **kwargs)
            for (key, cbk, ok_score, bad_score, args, kwargs) in self.pending_tasks
        )
        self.pending_tasks = []

        for key, result, failed, duration, ok_score, bad_score in results:
            self.add(key, result, failed, duration, ok_score, bad_score)

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
            "ok_score",
            "bad_score",
        ]
        output = pd.DataFrame([], columns=output_metrics)
        for metric in self.scores:
            values = self.scores[metric]["values"]
            errors = self.scores[metric]["errors"]
            ok_score = self.scores[metric]["ok_score"]
            bad_score = self.scores[metric]["bad_score"]
            durations = round(np.mean(self.scores[metric]["durations"]), 2)

            score_min = np.min(values)
            score_max = np.max(values)
            score_mean = np.mean(values)
            score_median = np.median(values)
            score_stddev = np.std(values)
            score_iqr = iqr(values)
            score_rounds = len(values)
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
                            ok_score,
                            bad_score,
                        ]
                    ],
                    columns=output_metrics,
                    index=[metric],
                )
            )

        return output
