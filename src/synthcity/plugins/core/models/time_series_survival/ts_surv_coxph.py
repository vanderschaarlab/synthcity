# stdlib
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.models.survival_analysis.surv_coxph import (
    CoxPHSurvivalAnalysis,
)
from synthcity.utils.constants import DEVICE
from synthcity.utils.reproducibility import enable_reproducible_results

# synthcity relative
from ._base import TimeSeriesSurvivalPlugin
from .ts_surv_dynamic_deephit import DynamicDeephitTimeSeriesSurvival


class CoxTimeSeriesSurvival(TimeSeriesSurvivalPlugin):
    def __init__(
        self,
        # prediction
        alpha: float = 0.05,
        penalizer: float = 0.1,
        device: Any = DEVICE,
        # embeddings
        emb_n_iter: int = 1000,
        emb_batch_size: int = 100,
        emb_lr: float = 1e-3,
        emb_n_layers_hidden: int = 1,
        emb_n_units_hidden: int = 40,
        emb_split: int = 100,
        emb_rnn_type: str = "GRU",
        emb_output_type: str = "MLP",
        emb_alpha: float = 0.34,
        emb_beta: float = 0.27,
        emb_sigma: float = 0.21,
        emb_dropout: float = 0.06,
        emb_patience: int = 20,
        # hyperopt helper
        n_iter: Optional[int] = None,
        random_state: int = 0,
    ) -> None:
        super().__init__()
        enable_reproducible_results(random_state)

        if n_iter is not None:
            emb_n_iter = n_iter

        self.emb_model = DynamicDeephitTimeSeriesSurvival(
            n_iter=emb_n_iter,
            batch_size=emb_batch_size,
            lr=emb_lr,
            n_layers_hidden=emb_n_layers_hidden,
            n_units_hidden=emb_n_units_hidden,
            split=emb_split,
            rnn_type=emb_rnn_type,
            output_type=emb_output_type,
            alpha=emb_alpha,
            beta=emb_beta,
            sigma=emb_sigma,
            dropout=emb_dropout,
            patience=emb_patience,
            random_state=random_state,
        )
        self.pred_model = CoxPHSurvivalAnalysis(
            alpha=alpha,
            penalizer=penalizer,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        static: Optional[np.ndarray],
        temporal: np.ndarray,
        observation_times: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
    ) -> TimeSeriesSurvivalPlugin:

        self.emb_model.fit(static, temporal, observation_times, T, E)
        embeddings = self.emb_model.predict_emb(
            static, temporal, observation_times
        ).reshape(len(T), -1)
        self.pred_model.fit(pd.DataFrame(embeddings), pd.Series(T), pd.Series(E))

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(
        self,
        static: Optional[np.ndarray],
        temporal: np.ndarray,
        observation_times: np.ndarray,
        time_horizons: List,
    ) -> np.ndarray:
        "Predict risk"

        embeddings = self.emb_model.predict_emb(
            static, temporal, observation_times
        ).reshape(len(temporal), -1)
        return self.pred_model.predict(pd.DataFrame(embeddings), time_horizons)

    @staticmethod
    def name() -> str:
        return "ts_coxph"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        return (
            CoxPHSurvivalAnalysis.hyperparameter_space()
            + DynamicDeephitTimeSeriesSurvival.hyperparameter_space(prefix="emb_")
        )
