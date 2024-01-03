# stdlib
import hashlib
import json
import platform
import random
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
from IPython.display import display
from pydantic import validate_arguments

# synthcity absolute
import synthcity.logger as log
from synthcity.benchmark.utils import augment_data, get_json_serializable_kwargs
from synthcity.metrics import Metrics
from synthcity.metrics.scores import ScoreEvaluator
from synthcity.plugins import Plugins
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.utils.reproducibility import clear_cache, enable_reproducible_results
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
        augmented_reuse_if_exists: bool = True,
        task_type: str = "classification",  # classification, regression, survival_analysis, time_series
        workspace: Path = Path("workspace"),
        augmentation_rule: str = "equal",
        strict_augmentation: bool = False,
        ad_hoc_augment_vals: Optional[Dict] = None,
        use_metric_cache: bool = True,
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
                    'privacy': ['delta-presence', 'k-anonymization', 'k-map', 'distinct l-diversity', 'identifiability_score', 'DomiasMIA_BNAF', 'DomiasMIA_KDE', 'DomiasMIA_prior']
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
                If the current synthetic dataset is cached, it will be reused for the experiments. Defaults to True.
            augmented_reuse_if_exists: bool
                If the current augmented dataset is cached, it will be reused for the experiments. Defaults to True.
            task_type: str
                The type of problem. Relevant for evaluating the downstream models with the correct metrics. Valid tasks are:  "classification", "regression", "survival_analysis", "time_series", "time_series_survival".
            workspace: Path
                Path for caching experiments. Default: "workspace".
            augmentation_rule: str
                The rule used to achieve the desired proportion records with each value in the fairness column. Possible values are: 'equal', 'log', and 'ad-hoc'. Defaults to "equal".
            strict_augmentation: bool
                Flag to ensure that the condition for generating synthetic data is strictly met. Defaults to False.
            ad_hoc_augment_vals: Dict
                A dictionary containing the number of each class to augment the real data with. This is only required if using the rule="ad-hoc" option. Defaults to None.
            use_metric_cache: bool
                If the current metric has been previously run and is cached, it will be reused for the experiments. Defaults to True.
            plugin_kwargs:
                Optional kwargs for each algorithm. Example {"adsgan": {"n_iter": 10}},
        """
        out = {}

        experiment_name = X.hash()

        workspace.mkdir(parents=True, exist_ok=True)

        plugin_cats = ["generic", "privacy", "domain_adaptation"]
        if X.type() == "images":
            plugin_cats.append("images")
        elif task_type == "survival_analysis":
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
                serializable_kwargs = get_json_serializable_kwargs(kwargs)
                kwargs_hash_raw = json.dumps(
                    serializable_kwargs, sort_keys=True
                ).encode()
                hash_object = hashlib.sha256(kwargs_hash_raw)
                kwargs_hash = hash_object.hexdigest()

            augmentation_arguments = {
                "augmentation_rule": augmentation_rule,
                "strict_augmentation": strict_augmentation,
                "ad_hoc_augment_vals": ad_hoc_augment_vals,
            }
            augmentation_arguments_hash_raw = json.dumps(
                copy(augmentation_arguments), sort_keys=True
            ).encode()
            augmentation_hash_object = hashlib.sha256(augmentation_arguments_hash_raw)
            augmentation_hash = augmentation_hash_object.hexdigest()

            repeats_list = list(range(repeats))
            random.shuffle(repeats_list)

            for repeat in repeats_list:
                enable_reproducible_results(repeat)

                kwargs["workspace"] = workspace
                kwargs["random_state"] = repeat

                clear_cache()

                X_syn_cache_file = (
                    workspace
                    / f"{experiment_name}_{testcase}_{plugin}_{kwargs_hash}_{platform.python_version()}_{repeat}.bkp"
                )
                X_ref_syn_cache_file = (
                    workspace
                    / f"{experiment_name}_{testcase}_{plugin}_{kwargs_hash}_{platform.python_version()}_{repeat}_reference.bkp"
                )
                generator_file = (
                    workspace
                    / f"{experiment_name}_{testcase}_{plugin}_{kwargs_hash}_{platform.python_version()}_generator_{repeat}.bkp"
                )
                X_augment_cache_file = (
                    workspace
                    / f"{experiment_name}_{testcase}_{plugin}_augmentation_{augmentation_hash}_{kwargs_hash}_{platform.python_version()}_{repeat}.bkp"
                )
                augment_generator_file = (
                    workspace
                    / f"{experiment_name}_{testcase}_{plugin}_augmentation_{augmentation_hash}_{kwargs_hash}_{platform.python_version()}_generator_{repeat}.bkp"
                )

                log.info(
                    f"[testcase] Experiment repeat: {repeat} task type: {task_type} Train df hash = {experiment_name}"
                )

                # TODO: caches should be from the same version of Synthcity. Different APIs will crash.
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

                if X_syn_cache_file.exists() and synthetic_reuse_if_exists:
                    X_syn = load_from_file(X_syn_cache_file)
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
                        save_to_file(X_syn_cache_file, X_syn)

                # X_ref_syn is the reference synthetic data used for DomiasMIA metrics
                if X_ref_syn_cache_file.exists() and synthetic_reuse_if_exists:
                    X_ref_syn = load_from_file(X_ref_syn_cache_file)
                else:
                    try:
                        X_ref_syn = generator.generate(
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
                        save_to_file(X_ref_syn_cache_file, X_ref_syn)

                # Augmentation
                if metrics and any(
                    "augmentation" in metric
                    for metric in [x for v in metrics.values() for x in v]
                ):
                    if augment_generator_file.exists() and augmented_reuse_if_exists:
                        augment_generator = load_from_file(augment_generator_file)
                    else:
                        augment_generator = Plugins(categories=plugin_cats).get(
                            plugin,
                            **kwargs,
                        )
                        try:
                            if not X.get_fairness_column():
                                raise ValueError(
                                    "To use the augmentation metrics, `fairness_column` must be set to a string representing the name of a column in the DataLoader."
                                )
                            augment_generator.fit(
                                X.train(),
                                cond=X.train()[X.get_fairness_column()],
                            )
                        except BaseException as e:
                            log.critical(
                                f"[{plugin}][take {repeat}] failed to fit augmentation generator: {e}"
                            )
                            continue
                        if synthetic_cache:
                            save_to_file(augment_generator_file, augment_generator)

                    if X_augment_cache_file.exists() and augmented_reuse_if_exists:
                        X_augmented = load_from_file(X_augment_cache_file)
                    else:
                        try:
                            X_augmented = augment_data(
                                X.train(),
                                augment_generator,
                                rule=augmentation_rule,
                                strict=strict_augmentation,
                                ad_hoc_augment_vals=ad_hoc_augment_vals,
                                **generate_kwargs,
                            )
                            if len(X_augmented) == 0:
                                raise RuntimeError("Plugin failed to generate data")
                        except BaseException as e:
                            log.critical(
                                f"[{plugin}][take {repeat}] failed to generate augmentation data: {e}"
                            )
                            continue
                        if synthetic_cache:
                            save_to_file(X_augment_cache_file, X_augmented)
                else:
                    X_augmented = None
                evaluation = Metrics.evaluate(
                    X_test if X_test is not None else X.test(),
                    X_syn,
                    X.train(),
                    X_ref_syn,
                    X_augmented,
                    metrics=metrics,
                    task_type=task_type,
                    workspace=workspace,
                    use_cache=use_metric_cache,
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
