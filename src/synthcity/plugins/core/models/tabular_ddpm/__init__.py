# stdlib
from copy import deepcopy
from typing import Any, Optional, Union

# third party
import numpy as np
import pandas as pd
import torch
from torch import nn
from pydantic import validate_arguments

# synthcity absolute
from synthcity.utils.constants import DEVICE
from synthcity.metrics.weighted_metrics import WeightedMetrics

from .gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion # noqa
from .modules import MLPDiffusion, ResNetDiffusion # noqa
from .utils import TensorDataLoader


class TabDDPM(nn.Module):
    
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_iter = 10000,
        lr = 0.002,
        weight_decay = 1e-4,
        batch_size = 1024,
        num_timesteps = 1000,
        gaussian_loss_type = 'mse',
        scheduler = 'cosine',
        device: Any = DEVICE,
        log_interval: int = 100,
        print_interval: int = 500,
        # model params
        model_type = 'mlp',
        rtdl_params: Optional[dict] = None,  # {'d_layers', 'dropout'}
        dim_label_emb: int = 128,
        # early stopping
        n_iter_min: int = 100,
        n_iter_print: int = 50,
        patience: int = 5,
    ) -> None:
        super().__init__()
        self.__dict__.update(locals())
        del self.self, self.kwargs
        
    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

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

    def fit(self, X: pd.DataFrame, cond=None, **kwargs: Any):
        if cond is not None:
            n_classes = len(np.unique(cond))
        else:
            n_classes = 0
            
        model_params = dict(
            num_classes=n_classes,
            is_y_cond=cond is not None,
            rtdl_params=self.rtdl_params,
            dim_t = self.dim_label_emb
        )
        
        tensors = [X] if cond is None else [X, cond]
        tensors = [torch.tensor(t.values, dtype=torch.float32, device=self.device) for t in tensors]
        self.dataloader = TensorDataLoader(tensors, batch_size=self.batch_size)

        self.diffusion = GaussianMultinomialDiffusion(
            model_type=self.model_type,
            model_params=model_params,
            num_numerical_features=self.encoder.n_features(),
            gaussian_loss_type=self.gaussian_loss_type,
            num_timesteps=self.num_timesteps,
            scheduler=self.scheduler,
            device=self.device
        ).to(self.device)
        
        self.ema_model = deepcopy(self.diffusion.denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.optimizer = torch.optim.AdamW(
            self.diffusion.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        
        for step, (x, y) in enumerate(self.dataloader):
            curr_loss_multi = 0.0
            curr_loss_gauss = 0.0
            curr_count = 0
            self.diffusion.train()
            
            self.optimizer.zero_grad()
            loss_multi, loss_gauss = self.diffusion.mixed_loss(x, dict(y=y))
            loss = loss_multi + loss_gauss
            loss.backward()
            self.optimizer.step()

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += loss_multi.item() * len(x)
            curr_loss_gauss += loss_gauss.item() * len(x)

            if (step + 1) % self.log_interval == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_interval == 0:
                    print(f'Step {(step + 1)}/{self.n_iter} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                self.loss_history.loc[len(self.loss_history)] = [
                    step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            self._update_ema(self.ema_model.parameters(), self.model.parameters())

            if step == self.n_iter - 1:
                break
            
        return self

    def generate(self, count: int, cond=None):
        self.diffusion.eval()
        sample, out_dict = self.diffusion.sample_all(count)
        return sample, out_dict['y']
