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
        target_column: Optional[str] = None,
        synthetic_size: Optional[int] = None,
        synthetic_constraints: Optional[Constraints] = None,
        plugin_kwargs: Dict = {},
    ) -> pd.DataFrame:
        """Benchmark the performance of several algorithms.

        Args:
            plugins:
                The list of algorithms to evaluate
            X:
                The baseline dataset to learn
            sensitive_columns:
                Optional list of sensitive columns, used for the privacy metrics.
            metrics:
                List of metrics to test. By default, all metrics are evaluated.
            repeats:
                Number of test repeats
            target_columns:
                The name of the column to use as target for benchmarking the performance metrics. By default, it uses the last column in the dataframe.
            synthetic_size: int
                The size of the synthetic dataset. By default, it is len(X).
            synthetic_constraints:
                Optional constraints on the synthetic data. By default, it inherits the constraints from X.
            plugin_kwargs:
                Optional kwargs for each algorithm. Example {"adsgan": {"n_iter": 10}},
        """
        out = {}
        for plugin in plugins:
            log.info(f"Benchmarking plugin : {plugin}")
            scores = ScoreEvaluator()

            kwargs = {}
            if plugin in plugin_kwargs:
                kwargs = plugin_kwargs[plugin]

            for repeat in range(repeats):
                log.info(f" Experiment repeat: {repeat}")
                generator = Plugins().get(plugin, **kwargs)

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
                    target_column=target_column,
                )

                mean_score = evaluation["mean"].to_dict()
                errors = evaluation["errors"].to_dict()
                duration = evaluation["durations"].to_dict()
                direction = evaluation["direction"].to_dict()

                for key in mean_score:
                    scores.add(
                        key,
                        mean_score[key],
                        errors[key],
                        duration[key],
                        direction[key],
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

            display(results[plugin].drop(columns=["direction"]))
            print()
