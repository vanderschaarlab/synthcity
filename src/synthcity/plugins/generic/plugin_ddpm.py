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
from synthcity.plugins.core.models.tabular_ddpm import GaussianMultinomialDiffusion, MLPDiffusion, ResNetDiffusion
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
        n_iter = 1000,
        lr = 0.002,
        weight_decay = 1e-4,
        batch_size = 1024,
        model_type = 'mlp',
        model_params = None,
        num_timesteps = 1000,
        gaussian_loss_type = 'mse',
        scheduler = 'cosine',
        change_val = False,
        device: Any = DEVICE,
        log_interval: int = 100,
        print_interval: int = 500,
        # early stopping
        n_iter_min: int = 100,
        n_iter_print: int = 50,
        patience: int = 5,
        patience_metric: Optional[WeightedMetrics] = None,
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

        if patience_metric is None:
            patience_metric = WeightedMetrics(
                metrics=[("detection", "detection_mlp")],
                weights=[1],
                workspace=workspace,
            )
            
        self.__dict__.update(locals())
        del self.self, self.kwargs

    @staticmethod
    def name() -> str:
        return "ddpm"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        raise NotImplementedError

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _one_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()
        return loss_multi, loss_gauss

    def _update_ema(self, target_params, source_params, rate=0.999):
        """
        Update target parameters to be closer to those of source parameters using
        an exponential moving average.
        :param target_params: the target parameter sequence.
        :param source_params: the source parameter sequence.
        :param rate: the EMA rate (closer to 1 means slower).
        """
        for targ, src in zip(target_params, source_params):
            targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "TabDDPMPlugin":
        # TODO: add parameters of TabularEncoder
        self.encoder = TabularEncoder().fit(X)
        
        if self.model_type == 'mlp':
            self.model = MLPDiffusion(**self.model_params)
        elif self.model_type == 'resnet':
            self.model = ResNetDiffusion(**self.model_params)
        else:
            raise "Unknown model!"
        
        self.diffusion = GaussianMultinomialDiffusion(
            denoise_fn=self.model,
            num_numerical_features=self.encoder.n_features(),
            gaussian_loss_type=self.gaussian_loss_type,
            num_timesteps=self.num_timesteps,
            scheduler=self.scheduler,
            device=self.device
        ).to(self.device)
        
        self.ema_model = deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.detach_()

        self.optimizer = torch.optim.AdamW(
            self.diffusion.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
        # TODO: check data type of X
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        for step in range(self.n_iter):
            x, out_dict = next(X)
            out_dict = {'y': out_dict}
            batch_loss_multi, batch_loss_gauss = self._one_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_interval == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_interval == 0:
                    print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                self.loss_history.loc[len(self.loss_history)] = [
                    step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            self._update_ema(self.ema_model.parameters(), self.model.parameters())

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> DataLoader:
        self.diffusion.eval()
        # TODO: check self.model.sample_all
        return self._safe_generate(self.diffusion.sample_all, count, syn_schema)


plugin = TabDDPMPlugin
