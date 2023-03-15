# stdlib
from abc import ABC, abstractmethod
from typing import Optional

# third party
import numpy as np
import pandas as pd
from torch import Tensor, nn

# synthcity absolute
from synthcity.metrics.weighted_metrics import WeightedMetrics


class Callback(ABC):
    """Abstract base class of callbacks."""

    @abstractmethod
    def on_epoch_begin(self, model: nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_epoch_end(self, model: nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_fit_begin(self, model: nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_fit_end(self, model: nn.Module) -> None:
        raise NotImplementedError


class EarlyStopping(Callback):
    def __init__(
        self,
        patience: int = 5,
        min_epochs: int = 100,
        patience_metric: Optional[WeightedMetrics] = None,
    ) -> None:
        self.patience = patience
        self.patience_metric = patience_metric
        self.min_epochs = min_epochs
        self.best_score = self._init_patience_score()
        self.best_model_state = None
        self.wait = 0
        self._epochs = 0

    def on_epoch_end(self, model: nn.Module) -> None:
        self._epochs += 1
        if self.patience_metric is not None:
            if not hasattr(self, "X_val"):
                self.X_val = model.X_val
                if isinstance(self.X_val, Tensor):
                    self.X_val = self.X_val.detach().cpu().numpy()
            self._evaluate_patience_metric(model)
        if self.wait >= self.patience and self._epochs >= self.min_epochs:
            raise StopIteration("Early stopping")

    def on_fit_end(self, model: nn.Module) -> None:
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)  # type: ignore

    def _init_patience_score(self) -> float:
        if self.patience_metric is None:
            return 0
        elif self.patience_metric.direction() == "minimize":
            return np.inf
        else:
            return -np.inf

    def _evaluate_patience_metric(self, model: nn.Module) -> None:
        X_val = self.X_val
        X_syn = model.generate(len(X_val))

        new_score = self.patience_metric.evaluate(  # type: ignore
            pd.DataFrame(X_val),
            pd.DataFrame(X_syn),
        )

        if self.patience_metric.direction() == "minimize":  # type: ignore
            is_new_best = new_score < self.best_score
        else:
            is_new_best = new_score > self.best_score

        if is_new_best:
            self.wait = 0
            self.best_score = new_score
            self.best_model_state = model.state_dict()
        else:
            self.wait += 1
