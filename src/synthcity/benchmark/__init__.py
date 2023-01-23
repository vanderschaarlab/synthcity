# stdlib
import hashlib
import json
import platform
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from pydantic import validate_arguments

# synthcity absolute
import synthcity.logger as log
from synthcity.metrics import Metrics
from synthcity.metrics.scores import ScoreEvaluator
from synthcity.plugins import Plugins
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.utils.reproducibility import enable_reproducible_results
from synthcity.utils.serialization import load_from_file, save_to_file


def print_score(mean: pd.Series, std: pd.Series) -> pd.Series:
    pd.options.mode.chained_assignment = None

    mean.loc[(mean < 1e-3) & (mean != 0)] = 1e-3
    std.loc[(std < 1e-3) & (std != 0)] = 1e-3

    mean = mean.round(3).astype(str)
    std = std.round(3).astype(str)

    return mean + " +/- " + std


class Benchmarks:
    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        tests: List[Tuple[str, str, dict]],  # test name, plugin name, plugin args
        X: DataLoader,
        X_test: Optional[DataLoader] = None,
        metrics: Optional[Dict] = None,
        repeats: int = 3,
        synthetic_size: Optional[int] = None,
        synthetic_constraints: Optional[Constraints] = None,
        synthetic_cache: bool = True,
        synthetic_reuse_if_exists: bool = True,
        task_type: str = "classification",  # classification, regression, survival_analysis, time_series
        workspace: Path = Path("workspace"),
        **generate_kwargs: Any,
    ) -> pd.DataFrame:
        """Benchmark the performance of several algorithms.

        Args:
            tests: List[Tuple[str, str, dict]]
                Tuples of form (testname: str, plugin_name, str, plugin_args: dict)
            X: DataLoader
                The baseline dataset to learn
            X_test: Optional[DataLoader]
                Optional test dataset for evaluation. If None, X will be split in train/test datasets.
            metrics:
                List of metrics to test. By default, all metrics are evaluated.
                Full dictionary of metrics is:
                {
                    'sanity': ['data_mismatch', 'common_rows_proportion', 'nearest_syn_neighbor_distance', 'close_values_probability', 'distant_values_probability'],
                    'stats': ['jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 'prdc', 'alpha_precision', 'survival_km_distance'],
                    'performance': ['linear_model', 'mlp', 'xgb', 'feat_rank_distance'],
                    'detection': ['detection_xgb', 'detection_mlp', 'detection_gmm', 'detection_linear'],
                    'privacy': ['delta-presence', 'k-anonymization', 'k-map', 'distinct l-diversity', 'identifiability_score']
                }
            repeats:
                Number of test repeats
            synthetic_size: int
                The size of the synthetic dataset. By default, it is len(X).
            synthetic_constraints:
                Optional constraints on the synthetic data. By default, it inherits the constraints from X.
            synthetic_cache: bool
                Enable experiment caching
            synthetic_reuse_if_exists: bool
                If the current synthetic dataset is cached, it will be reused for the experiments.
            task_type: str
                The type of problem. Relevant for evaluating the downstream models with the correct metrics. Valid tasks are:  "classification", "regression", "survival_analysis", "time_series", "time_series_survival".
            workspace: Path
                Path for caching experiments. Default: "workspace".
            plugin_kwargs:
                Optional kwargs for each algorithm. Example {"adsgan": {"n_iter": 10}},
        """
        out = {}

        experiment_name = X.hash()

        workspace.mkdir(parents=True, exist_ok=True)

        plugin_cats = ["generic", "privacy"]
        if task_type == "survival_analysis":
            plugin_cats.append("survival_analysis")
        elif task_type == "time_series" or task_type == "time_series_survival":
            plugin_cats.append("time_series")

        for testcase, plugin, kwargs in tests:
            log.info(f"Testcase : {testcase}")
            if not isinstance(kwargs, dict):
                raise ValueError(f"'kwargs' must be a dict for {testcase}:{plugin}")

            scores = ScoreEvaluator()

            kwargs_hash = ""
            if len(kwargs) > 0:
                kwargs_hash_raw = json.dumps(kwargs, sort_keys=True).encode()
                hash_object = hashlib.sha256(kwargs_hash_raw)
                kwargs_hash = hash_object.hexdigest()

            repeats_list = list(range(repeats))
            random.shuffle(repeats_list)

            for repeat in repeats_list:
                enable_reproducible_results(repeat)

                kwargs["workspace"] = workspace
                kwargs["random_state"] = repeat

                torch.cuda.empty_cache()

                cache_file = (
                    workspace
                    / f"{experiment_name}_{testcase}_{plugin}_{kwargs_hash}_{platform.python_version()}_{repeat}.bkp"
                )
                generator_file = (
                    workspace
                    / f"{experiment_name}_{testcase}_{plugin}_{kwargs_hash}_{platform.python_version()}_generator_{repeat}.bkp"
                )

                log.info(
                    f"[testcase] Experiment repeat: {repeat} task type: {task_type} Train df hash = {experiment_name}"
                )

                if generator_file.exists() and synthetic_reuse_if_exists:
                    generator = load_from_file(generator_file)
                else:
                    generator = Plugins(categories=plugin_cats).get(
                        plugin,
                        **kwargs,
                    )

                    generator.fit(X.train())

                    if synthetic_cache:
                        save_to_file(generator_file, generator)

                if cache_file.exists() and synthetic_reuse_if_exists:
                    X_syn = load_from_file(cache_file)
                else:
                    try:
                        X_syn = generator.generate(
                            count=synthetic_size,
                            constraints=synthetic_constraints,
                            **generate_kwargs,
                        )
                        if len(X_syn) == 0:
                            raise RuntimeError("Plugin failed to generate data")
                    except BaseException as e:
                        log.critical(f"[{plugin}][take {repeat}] failed: {e}")
                        continue

                    if synthetic_cache:
                        save_to_file(cache_file, X_syn)

                evaluation = Metrics.evaluate(
                    X_test if X_test is not None else X,
                    X_syn,
                    metrics=metrics,
                    task_type=task_type,
                    workspace=workspace,
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
            out[testcase] = scores.to_dataframe()

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
            mean = results[plugin]["mean"]
            stddev = results[plugin]["stddev"]
            means.append(print_score(mean, stddev))

        avg = pd.concat(means, axis=1)
        avg = avg.set_axis(results.keys(), axis=1)

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

    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def highlight(
        results: Dict,
    ) -> None:
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        means = []
        for plugin in results:
            data = results[plugin]["mean"]
            directions = results[plugin]["direction"].to_dict()
            means.append(data)

        out = pd.concat(means, axis=1)
        out.set_axis(results.keys(), axis=1, inplace=True)

        bad_highlight = "background-color: lightcoral;"
        ok_highlight = "background-color: green;"
        default = ""

        def highlights(row: pd.Series) -> Any:
            metric = row.name
            if directions[metric] == "minimize":
                best_val = np.min(row.values)
                worst_val = np.max(row)
            else:
                best_val = np.max(row.values)
                worst_val = np.min(row)

            styles = []
            for val in row.values:
                if val == best_val:
                    styles.append(ok_highlight)
                elif val == worst_val:
                    styles.append(bad_highlight)
                else:
                    styles.append(default)

            return styles

        out.style.apply(highlights, axis=1)

        return out
