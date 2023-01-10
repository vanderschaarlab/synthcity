# stdlib
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.models.survival_analysis.surv_xgb import XGBSurvivalAnalysis
from synthcity.utils.constants import DEVICE
from synthcity.utils.reproducibility import enable_reproducible_results

# synthcity relative
from ._base import TimeSeriesSurvivalPlugin
from .ts_surv_dynamic_deephit import DynamicDeephitTimeSeriesSurvival


class XGBTimeSeriesSurvival(TimeSeriesSurvivalPlugin):
    def __init__(
        self,
        # prediction
        n_estimators: int = 100,
        colsample_bynode: float = 0.5,
        max_depth: int = 5,
        subsample: float = 0.5,
        learning_rate: float = 5e-2,
        min_child_weight: int = 50,
        tree_method: str = "hist",
        booster: int = 0,
        random_state: int = 0,
        objective: str = "aft",  # "aft", "cox"
        strategy: str = "km",  # "weibull", "debiased_bce", "km"
        bce_n_iter: int = 1000,
        time_points: int = 100,
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
        **kwargs: Any,
    ) -> None:
        super().__init__()
        enable_reproducible_results(random_state)

        if n_iter is not None:
            n_estimators = n_iter
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
        self.pred_model = XGBSurvivalAnalysis(
            n_estimators=n_estimators,
            colsample_bynode=colsample_bynode,
            max_depth=max_depth,
            subsample=subsample,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            tree_method=tree_method,
            booster=booster,
            random_state=random_state,
            objective=objective,
            strategy=strategy,
            time_points=time_points,
            bce_n_iter=bce_n_iter,
            device=device,
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
        embeddings = self.emb_model.predict_emb(static, temporal, observation_times)
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

        embeddings = self.emb_model.predict_emb(static, temporal, observation_times)
        return self.pred_model.predict(pd.DataFrame(embeddings), time_horizons)

    @staticmethod
    def name() -> str:
        return "ts_xgb"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        return (
            XGBSurvivalAnalysis.hyperparameter_space()
            + DynamicDeephitTimeSeriesSurvival.hyperparameter_space(prefix="emb_")
        )
