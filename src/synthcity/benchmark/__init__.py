# stdlib
from typing import Dict, List, Optional

# third party
import pandas as pd
from IPython.display import display
from pydantic import validate_arguments

# synthcity absolute
import synthcity.logger as log
from synthcity.metrics import Metrics
from synthcity.metrics.scores import ScoreEvaluator
from synthcity.plugins import Plugins
from synthcity.plugins.core.constraints import Constraints


class Benchmarks:
    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        plugins: List,
        X: pd.DataFrame,
        sensitive_columns: List[str] = [],
        metrics: Optional[Dict] = None,
        repeats: int = 3,
        synthetic_size: Optional[int] = None,
        synthetic_constraints: Optional[Constraints] = None,
    ) -> pd.DataFrame:
        out = {}
        for plugin in plugins:
            log.info(f"Benchmarking plugin : {plugin}")
            scores = ScoreEvaluator()
            for repeat in range(repeats):
                log.info(f" Experiment repeat: {repeat}")
                generator = Plugins().get(plugin)

                try:
                    generator.fit(X)
                    X_syn = generator.generate(
                        count=synthetic_size, constraints=synthetic_constraints
                    )
                    if len(X_syn) == 0:
                        raise RuntimeError("Plugin failed to generate data")
                except BaseException as e:
                    log.critical(f"[{plugin}][take {repeat}] failed: {e}")
                    continue

                evaluation = Metrics.evaluate(
                    X,
                    X_syn,
                    sensitive_columns=sensitive_columns,
                    metrics=metrics,
                )

                mean_score = evaluation["mean"].to_dict()
                errors = evaluation["errors"].to_dict()
                duration = evaluation["durations"].to_dict()
                ok_score = evaluation["ok_score"].to_dict()
                bad_score = evaluation["bad_score"].to_dict()

                for key in mean_score:
                    scores.add(
                        key,
                        mean_score[key],
                        errors[key],
                        duration[key],
                        ok_score[key],
                        bad_score[key],
                    )
            out[plugin] = scores.to_dataframe()

        return out

    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def print(
        results: Dict,
    ) -> None:
        pd.set_option("display.max_rows", None, "display.max_columns", None)

        means = []
        for plugin in results:
            data = results[plugin]["mean"]
            means.append(data)

        avg = pd.concat(means, axis=1)
        avg.set_axis(results.keys(), axis=1, inplace=True)

        if len(means) > 1:
            print()
            print("\033[4m" + "\033[1m" + "Comparatives" + "\033[0m" + "\033[0m")
            display(avg)

        for plugin in results:
            print()
            print("\033[4m" + "\033[1m" + f"Plugin : {plugin}" + "\033[0m" + "\033[0m")

            display(results[plugin].drop(columns=["ok_score", "bad_score"]))
            print()
