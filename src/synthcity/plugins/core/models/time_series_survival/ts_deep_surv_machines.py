# stdlib
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd
from auton_survival.models.dsm import DeepRecurrentSurvivalMachines
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    IntegerDistribution,
)
from synthcity.utils.constants import DEVICE
from synthcity.utils.reproducibility import enable_reproducible_results

# synthcity relative
from ._base import TimeSeriesSurvivalPlugin


class DeepSurvivalMachinesTimeSeriesSurvival(TimeSeriesSurvivalPlugin):
    """
    Args:
        k: int
            The number of underlying parametric distributions.
        distribution: str
            Choice of the underlying survival distributions. One of 'Weibull', 'LogNormal'. Default is 'Weibull'.
        temp: float
            The logits for the gate are rescaled with this value. Default is 1000.
        discount: float
            a float in [0,1] that determines how to discount the tail bias from the uncensored instances. Default is 1.

    """

    def __init__(
        self,
        k: int = 3,  # The number of underlying parametric distributions.
        distribution: str = "Weibull",  # Weibull, LogNormal
        temp: float = 1000,  # The logits for the gate are rescaled with this value. Default is 1000.
        discount: float = 1.0,
        n_iter: int = 1000,
        batch_size: int = 100,
        lr: float = 1e-3,
        n_layers_hidden: int = 1,
        n_units_hidden: int = 100,
        seed: int = 0,
        device: Any = DEVICE,
        **kwargs: Any
    ) -> None:
        super().__init__()
        enable_reproducible_results(seed)

        self.lr = lr
        self.batch_size = batch_size
        self.n_iter = n_iter

        self.model = DeepRecurrentSurvivalMachines(
            layers=n_layers_hidden,
            hidden=n_units_hidden,
            k=k,
            distribution=distribution,
            temp=temp,
            discount=discount,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        static: Optional[np.ndarray],
        temporal: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
    ) -> TimeSeriesSurvivalPlugin:

        self.model.fit(
            temporal,
            T,
            E,
            batch_size=self.batch_size,
            learning_rate=self.lr,
            iters=self.n_iter,
        )
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(
        self,
        static: Optional[np.ndarray],
        temporal: np.ndarray,
        time_horizons: List,
    ) -> np.ndarray:
        "Predict risk"

        return pd.DataFrame(
            self.model.predict_risk(temporal, time_horizons), columns=time_horizons
        )

    @staticmethod
    def name() -> str:
        return "deep_survival_machines"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="n_units_hidden", low=10, high=100, step=10),
            IntegerDistribution(name="n_layers_hidden", low=1, high=4),
            CategoricalDistribution(name="batch_size", choices=[100, 200, 500]),
            CategoricalDistribution(name="lr", choices=[1e-2, 1e-3, 1e-4]),
        ]
