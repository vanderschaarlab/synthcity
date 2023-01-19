# stdlib
from pathlib import Path
from typing import Any, Optional, Tuple, Type

# third party
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split

# synthcity absolute
import synthcity.logger as log
from synthcity.metrics.eval_detection import SyntheticDetectionMLP
from synthcity.utils.redis_wrapper import RedisBackend
from synthcity.utils.serialization import (
    dataframe_cols_hash,
    load_from_file,
    save_to_file,
)

threshold = 10


def search_parameters(
    model_template: Type,
    X: pd.DataFrame,
    n_trials: int = 50,
    timeout: int = 6000,
    n_iter_min: int = 1000,
    random_state: int = 0,
    fail_score: int = 99,
    dry_run: bool = False,
    workspace: Path = Path("workspace"),
    predefined_params: dict = {},
) -> Optional[dict]:
    direction = "minimize"
    metric = "detection_mlp"

    search_len = min(len(X), 10000)
    X_target_train, X_target_test = train_test_split(
        X.sample(search_len, random_state=random_state), random_state=random_state
    )

    experiment_name = dataframe_cols_hash(X)
    study_name = f"hpo_tl_{model_template.name()}_{experiment_name}_metric_{metric}"
    study, pruner = create_study(
        study_name=study_name,
        direction=direction,
    )

    def evaluate_args(**kwargs: Any) -> float:
        kwargs["random_state"] = random_state
        kwargs["n_iter"] = n_iter_min

        for key in predefined_params:
            kwargs[key] = predefined_params[key]

        model = model_template(**kwargs)
        log.info(f"[HPO] Evaluate {model_template.name()} for {kwargs}")

        try:
            model.fit(X_target_train)

            X_fake = model.generate(len(X_target_test))
        except BaseException:
            return fail_score

        score = SyntheticDetectionMLP().evaluate(
            X_target_test,
            X_fake,
        )

        log.info(f"[HPO] Trial {kwargs}: score {score}")
        return score

    baseline_score_bkp = workspace / f"baseline_score_{study_name}"
    if baseline_score_bkp.exists():
        log.info(f"Use cached baseline for {study_name}")
        baseline_score = load_from_file(baseline_score_bkp)
    else:
        log.info(f"Evaluate baseline for {study_name}")
        baseline_score = evaluate_args()
        save_to_file(baseline_score_bkp, baseline_score)

    if len(model_template.hyperparameter_space()) == 0:
        return {}

    try:
        if dry_run:
            if baseline_score < study.best_value:
                return {}

            return study.best_trial.params
    except BaseException:
        pass

    def objective(trial: optuna.Trial) -> float:
        args = model_template.sample_hyperparameters_optuna(trial)
        pruner.check_trial(trial)

        score = evaluate_args(**args)
        pruner.report_score(score)

        return score

    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
    except EarlyStoppingExceeded:
        log.info("[HPO] Early stopping triggered for search")

    log.info(
        f"[HPO] Best trial for estimator {model_template.name()}: {study.best_value} for {study.best_trial.params}"
    )

    if baseline_score < study.best_value:
        return {}

    if fail_score == study.best_value:
        return None

    return study.best_trial.params


class EarlyStoppingExceeded(optuna.exceptions.OptunaError):
    pass


class ParamRepeatPruner:
    """Prunes reapeated trials, which means trials with the same paramters won't waste time/resources."""

    def __init__(
        self,
        study: optuna.study.Study,
        patience: int,
    ) -> None:
        self.study = study
        self.seen: set = set()

        self.best_score: float = -1
        self.no_improvement_for = 0
        self.patience = patience

        if self.study is not None:
            self.register_existing_trials()

    def register_existing_trials(self) -> None:
        for trial_idx, trial_past in enumerate(
            self.study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
        ):
            if trial_past.values[0] > self.best_score:
                self.best_score = trial_past.values[0]
                self.no_improvement_for = 0
            else:
                self.no_improvement_for += 1
            self.seen.add(hash(frozenset(trial_past.params.items())))

    def check_patience(
        self,
        trial: optuna.trial.Trial,
    ) -> None:
        if self.no_improvement_for > self.patience:
            raise EarlyStoppingExceeded()

    def check_trial(
        self,
        trial: optuna.trial.Trial,
    ) -> None:
        self.check_patience(trial)

        params = frozenset(trial.params.items())

        current_val = hash(params)
        if current_val in self.seen:
            raise optuna.exceptions.TrialPruned()

        self.seen.add(current_val)

    def report_score(self, score: float) -> None:
        if score > self.best_score:
            self.best_score = score
            self.no_improvement_for = 0
        else:
            self.no_improvement_for += 1


def create_study(
    study_name: str,
    direction: str = "maximize",
    load_if_exists: bool = True,
    storage_type: str = "redis",
    patience: int = threshold,
) -> Tuple[optuna.Study, ParamRepeatPruner]:
    """Helper for creating a new study.

    Args:
        study_name: str
            Study ID
        direction: str
            maximize/minimize
        load_if_exists: bool
            If True, it tries to load previous trials from the storage.
        storage_type: str
            redis/none
        patience: int
            How many trials without improvement to accept.

    """

    storage_obj = None
    if storage_type == "redis":
        try:
            backend = RedisBackend()
            storage_obj = backend.optuna()
        except BaseException:
            log.error("Failed to load Redis backed.")
            storage_obj = None

    try:
        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage_obj,
            load_if_exists=load_if_exists,
        )
    except BaseException as e:
        log.debug(f"create_study failed {e}")
        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
        )

    return study, ParamRepeatPruner(study, patience=patience)
