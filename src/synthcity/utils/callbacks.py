# stdlib
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

# third party
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn

# synthcity absolute
from synthcity.logger import info, warning
from synthcity.metrics.weighted_metrics import WeightedMetrics


class Callback(ABC):
    """Abstract base class of a plugin callback."""

    @abstractmethod
    def on_epoch_begin(self, model: Any) -> None:
        """Called at the beginning of each epoch."""

    @abstractmethod
    def on_epoch_end(self, model: Any) -> None:
        """Called at the end of each epoch."""

    @abstractmethod
    def on_fit_begin(self, model: Any) -> None:
        """Called at the beginning of fitting."""

    @abstractmethod
    def on_fit_end(self, model: Any) -> None:
        """Called at the end of fitting."""


class CallbackHookMixin(ABC):
    def __init__(self, callbacks: Sequence[Callback]) -> None:
        self.callbacks = callbacks

    def on_epoch_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(self)

    def on_epoch_end(self) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(self)

    def on_fit_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_begin(self)

    def on_fit_end(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_end(self)


class ValidationMixin(CallbackHookMixin):
    def __init__(
        self,
        valid_metric: WeightedMetrics,
        valid_size: float = 0,
        callbacks: Sequence[Callback] = (),
    ) -> None:
        super().__init__(callbacks)
        self.valid_metric = valid_metric
        self.valid_size = valid_size
        self.valid_set = None
        self.valid_score = None
        self.should_stop = False

    @property
    def metric_direction(self) -> str:
        return self.valid_metric.direction()

    def _set_val_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.valid_size > 0 and self.valid_metric is not None:
            data, self.valid_set = train_test_split(data, test_size=self.valid_size)
        return data

    @abstractmethod
    def generate(self, count: int, cond: Any = None) -> pd.DataFrame:
        """Generate synthetic data."""

    def validate(self) -> Optional[pd.DataFrame]:
        """Validate synthetic data."""
        if self.valid_set is None:
            warning("No validation set provided. Skipped validation.")
            return None
        syn_data = pd.DataFrame(self.generate(len(self.valid_set)))  # type: ignore
        return self.valid_metric.evaluate(self.valid_set, syn_data)

    def on_epoch_begin(self) -> None:
        self.valid_score = None
        self.should_stop = False
        super().on_epoch_begin()

    def on_epoch_end(self) -> None:
        if self.valid_set is not None:
            self.valid_score = self.validate()  # type: ignore
        super().on_epoch_end()


class EarlyStopping(Callback):
    def __init__(
        self,
        patience: int = 5,
        min_epochs: int = 100,
    ) -> None:
        self.patience = patience
        self.min_epochs = min_epochs
        self.best_score = None
        self.best_model_state = None
        self.best_epoch = None
        self.wait = 0
        self._epochs = 0

    def on_fit_begin(self, model: ValidationMixin) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("Model must be a PyTorch module.")

    def on_epoch_begin(self, model: ValidationMixin) -> None:
        pass

    def on_epoch_end(self, model: ValidationMixin) -> None:
        self._evaluate_patience_metric(model)
        self._epochs += 1
        if self.wait >= self.patience and self._epochs >= self.min_epochs:
            model.should_stop = True

    def on_fit_end(self, model: ValidationMixin) -> None:
        self._load_best_model(model)

    def _evaluate_patience_metric(self, model: ValidationMixin) -> None:
        new_score = model.valid_score

        if self.best_score is None:
            is_new_best = True
        elif model.metric_direction == "minimize":  # type: ignore
            is_new_best = new_score < self.best_score
        else:
            is_new_best = new_score > self.best_score

        if is_new_best:
            self.wait = 0
            self.best_score = new_score
            self.best_epoch = self._epochs  # type: ignore
            self._save_best_model(model)
        else:
            self.wait += 1

    def _load_best_model(self, model: ValidationMixin) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("Model must be a PyTorch module.")
        if self.best_model_state is not None:
            info(f"Loading best model from epoch {self.best_epoch}.")  # type: ignore
            model.load_state_dict(self.best_model_state)

    def _save_best_model(self, model: ValidationMixin) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("Model must be a PyTorch module.")
        self.best_model_state = model.state_dict()


class OptunaPruning(Callback):
    def __init__(self, trial: optuna.Trial) -> None:
        self.trial = trial
        self._steps = 0

    def on_epoch_end(self, model: ValidationMixin) -> None:
        self.trial.report(model.valid_score, self._steps)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        self._steps += 1
