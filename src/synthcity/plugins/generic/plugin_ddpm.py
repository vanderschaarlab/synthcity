"""
Reference: Kotelnikov, Akim et al. “TabDDPM: Modelling Tabular Data with Diffusion Models.” ArXiv abs/2209.15421 (2022): n. pag.
"""
# mypy: disable-error-code=override
# flake8: noqa: F401

# stdlib
from pathlib import Path
from typing import Any, List, Optional, Union

# third party
import numpy as np
import pandas as pd

# Necessary packages
from pydantic import validate_arguments

# synthcity absolute
from synthcity.metrics.weighted_metrics import WeightedMetrics
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.models.tabular_ddpm import TabDDPM
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE


class TabDDPMPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.generic.plugin_tab_ddpm.TabDDPMPlugin
        :parts: 1


    Tabular denoising diffusion probabilistic model.

    Args:
        ...

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from synthcity.plugins import Plugins
        >>> X, y = load_iris(as_frame = True, return_X_y = True)
        >>> X["target"] = y
        >>> plugin = Plugins().get("ddpm", n_iter = 100)
        >>> plugin.fit(X)
        >>> plugin.generate(50)

    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        *,
        is_classification: bool = False,
        n_iter: int = 1000,
        lr: float = 0.002,
        weight_decay: float = 1e-4,
        batch_size: int = 1024,
        model_type: str = "mlp",
        num_timesteps: int = 1000,
        gaussian_loss_type: str = "mse",
        scheduler: str = "cosine",
        device: Any = DEVICE,
        verbose: int = 0,
        log_interval: int = 100,
        print_interval: int = 500,
        # model params
        num_layers: int = 3,
        dim_hidden: int = 256,
        dropout: float = 0.0,
        dim_label_emb: int = 128,
        # early stopping
        n_iter_min: int = 100,
        n_iter_print: int = 50,
        patience: int = 5,
        # patience_metric: Optional[WeightedMetrics] = None,
        # core plugin arguments
        random_state: int = 0,
        workspace: Path = Path("workspace"),
        compress_dataset: bool = False,
        sampling_patience: int = 500,
        **kwargs: Any
    ) -> None:
        super().__init__(
            device=device,
            random_state=random_state,
            sampling_patience=sampling_patience,
            workspace=workspace,
            compress_dataset=compress_dataset,
            **kwargs
        )

        self.is_classification = is_classification

        rtdl_params = dict(d_layers=[dim_hidden] * num_layers, dropout=dropout)
        self.model = TabDDPM(
            n_iter=n_iter,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            num_timesteps=num_timesteps,
            gaussian_loss_type=gaussian_loss_type,
            scheduler=scheduler,
            device=device,
            verbose=verbose,
            log_interval=log_interval,
            print_interval=print_interval,
            model_type=model_type,
            rtdl_params=rtdl_params,
            dim_label_emb=dim_label_emb,
            n_iter_min=n_iter_min,
            n_iter_print=n_iter_print,
            patience=patience,
        )

    @staticmethod
    def name() -> str:
        return "ddpm"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        """
        Hyperparameter Search space (from the paper)
        ----------------------------------------------
        Learning rate           LogUniform[0.00001, 0.003]
        Batch size              Cat{256, 4096}
        Diffusion timesteps     Cat{100, 1000}
        Training iterations     Cat{5000, 10000, 20000}
        Number of MLP layers    Int{2, 4, 6, 8}
        MLP width of layers     Int{128, 256, 512, 1024}
        Proportion of samples   Float{0.25, 0.5, 1, 2, 4, 8}
        ----------------------------------------------
        Dropout                 0.0
        Scheduler               cosine (Nichol, 2021)
        Gaussian diffusion loss MSE
        """
        return [
            # TODO: change to loguniform distribution
            CategoricalDistribution(name="lr", choices=[1e-5, 1e-4, 1e-3, 2e-3, 3e-3]),
            CategoricalDistribution(name="batch_size", choices=[256, 4096]),
            CategoricalDistribution(name="num_timesteps", choices=[100, 1000]),
            CategoricalDistribution(name="n_iter", choices=[5000, 10000, 20000]),
            CategoricalDistribution(name="num_layers", choices=[2, 4, 6, 8]),
            CategoricalDistribution(name="dim_hidden", choices=[128, 256, 512, 1024]),
        ]

    def _fit(
        self, data: DataLoader, cond: Any = None, **kwargs: Any
    ) -> "TabDDPMPlugin":
        if self.is_classification:
            if cond is not None:
                raise ValueError(
                    "cond is already given by the labels for classification"
                )
            _, cond = data.unpack()
            self._labels, self._cond_dist = np.unique(cond, return_counts=True)
            self._cond_dist = self._cond_dist / self._cond_dist.sum()

        # NOTE: should we include the target column in `df`?
        df = data.dataframe()

        if cond is not None:
            cond = pd.Series(cond, index=df.index)

        # self.encoder = TabularEncoder().fit(X)

        self.model.fit(df, cond, **kwargs)

        return self

    def _generate(
        self, count: int, syn_schema: Schema, cond: Any = None, **kwargs: Any
    ) -> DataLoader:
        if self.is_classification and cond is None:
            # randomly generate labels following the distribution of the training data
            cond = np.random.choice(self._labels, size=count, p=self._cond_dist)

        def callback(count, cond=cond):  # type: ignore
            return self.model.generate(count, cond=cond)

        return self._safe_generate(callback, count, syn_schema, **kwargs)


plugin = TabDDPMPlugin
