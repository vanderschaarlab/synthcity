# stdlib
import json
from pathlib import Path
from typing import Dict, List, Optional

# third party
import pandas as pd
from IPython.display import display
from pydantic import validate_arguments
from sklearn.model_selection import train_test_split

# synthcity absolute
import synthcity.logger as log
from synthcity.metrics import Metrics
from synthcity.metrics.scores import ScoreEvaluator
from synthcity.plugins import Plugins
from synthcity.plugins.core.constraints import Constraints
from synthcity.utils.reproducibility import enable_reproducible_results
from synthcity.utils.serialization import dataframe_hash, load_from_file, save_to_file


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
        synthetic_cache: bool = True,
        synthetic_reuse_is_exists: bool = True,
        task_type: str = "classification",  # classification, regression, survival_analysis
        target_column: Optional[str] = None,
        time_to_event_column: Optional[
            str
        ] = None,  # only for task_type = survival_analysis
        time_horizons: Optional[List] = None,  # only for task_type = survival_analysis
        train_size: float = 0.8,
        workspace: Path = Path("workspace"),
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
            synthetic_size: int
                The size of the synthetic dataset. By default, it is len(X).
            synthetic_constraints:
                Optional constraints on the synthetic data. By default, it inherits the constraints from X.
            synthetic_cache: bool
                Enable experiment caching
            synthetic_reuse_is_exists: bool
                If the current synthetic dataset is cached, it will be reused for the experiments.
            task_type: str
                The task type to benchmark for performance. Options: classification, regression, survival_analysis.
            target_column:
                The name of the column to use as target for benchmarking the performance metrics. By default, it uses the last column in the dataframe.
            time_to_event_column: Optional str.
                Only for survival_analysis: which column to use for time to event.
            time_horizons: Optional list
                Only for survival_analysis: which time horizons to use for performance evaluation.
            workspace: Path
                Path for caching experiments
            plugin_kwargs:
                Optional kwargs for each algorithm. Example {"adsgan": {"n_iter": 10}},
        """
        out = {}

        experiment_name = dataframe_hash(X)
        workspace.mkdir(parents=True, exist_ok=True)

        plugin_cats = ["generic"]
        if task_type == "survival_analysis":
            plugin_cats.append("survival_analysis")

        for plugin in plugins:
            log.info(f"Benchmarking plugin : {plugin}")
            scores = ScoreEvaluator()

            kwargs = {}
            kwargs_hash = ""
            if plugin in plugin_kwargs:
                kwargs = plugin_kwargs[plugin]
            if len(kwargs) > 0:
                kwargs_hash = json.dumps(kwargs, sort_keys=True)

            for repeat in range(repeats):
                enable_reproducible_results(repeat)
                cache_file = (
                    workspace / f"{experiment_name}_{plugin}{kwargs_hash}_{repeat}.bkp"
                )

                X_train, X_test = train_test_split(
                    X, train_size=train_size, random_state=repeat
                )

                log.info(
                    f" Experiment repeat: {repeat} task type: {task_type} Train df hash = {dataframe_hash(X_train)}"
                )

                if cache_file.exists() and synthetic_reuse_is_exists:
                    X_syn = load_from_file(cache_file)
                else:
                    generator = Plugins(categories=plugin_cats).get(
                        plugin,
                        **kwargs,
                        target_column=target_column,
                        time_to_event_column=time_to_event_column,
                        time_horizons=time_horizons,
                    )

                    try:
                        generator.fit(X_train)
                        X_syn = generator.generate(
                            count=synthetic_size, constraints=synthetic_constraints
                        )
                        if len(X_syn) == 0:
                            raise RuntimeError("Plugin failed to generate data")
                    except BaseException as e:
                        log.critical(f"[{plugin}][take {repeat}] failed: {e}")
                        continue

                    if synthetic_cache:
                        save_to_file(cache_file, X_syn)

                evaluation = Metrics.evaluate(
                    X_train,
                    X_test,
                    X_syn,
                    sensitive_columns=sensitive_columns,
                    metrics=metrics,
                    task_type=task_type,
                    target_column=target_column,
                    time_to_event_column=time_to_event_column,
                    time_horizons=time_horizons,
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
        only_comparatives: bool = True,
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

            if only_comparatives:
                return

        for plugin in results:
            print()
            print("\033[4m" + "\033[1m" + f"Plugin : {plugin}" + "\033[0m" + "\033[0m")

            display(results[plugin].drop(columns=["direction"]))
            print()
