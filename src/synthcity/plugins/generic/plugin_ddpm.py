"""
Reference: Kotelnikov, Akim et al. “TabDDPM: Modelling Tabular Data with Diffusion Models.” ArXiv abs/2209.15421 (2022): n. pag.
"""

# stdlib
from pathlib import Path
from copy import deepcopy
from typing import Any, List, Optional, Union

# third party
import numpy as np
import pandas as pd

# Necessary packages
from pydantic import validate_arguments
import torch
from torch.utils.data import sampler

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
from synthcity.plugins.core.models.tabular_encoder import TabularEncoder
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
        n_iter = 1000,
        lr = 0.002,
        weight_decay = 1e-4,
        batch_size = 1024,
        model_type = 'mlp',
        num_timesteps = 1000,
        gaussian_loss_type = 'mse',
        scheduler = 'cosine',
        device: Any = DEVICE,
        log_interval: int = 100,
        print_interval: int = 500,
        # model params
        rtdl_params: Optional[dict] = None,  # {'d_layers', 'dropout'}
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

        self.model = TabDDPM(
            n_iter=n_iter,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            num_timesteps=num_timesteps,
            gaussian_loss_type=gaussian_loss_type,
            scheduler=scheduler,
            device=device, 
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
        raise NotImplementedError

    def _fit(self, data: DataLoader, cond: pd.Series = None, **kwargs) -> "TabDDPMPlugin":
        if self.is_classification:
            assert cond is None
            _, cond = data.unpack()
            
        if cond is not None:
            cond = pd.Series(cond, index=data.index)
        data = data.dataframe()
            
        # self.encoder = TabularEncoder().fit(X)
        
        self.model.fit(data, cond, **kwargs)
        
    def _generate(self, count: int, syn_schema: Schema, cond=None, **kwargs: Any) -> DataLoader:
        def callback(count, cond):
            sample, cond = self.model.generate(count, cond=cond)
            return sample
        return self._safe_generate(callback, count, syn_schema, cond=cond, **kwargs)

plugin = TabDDPMPlugin
