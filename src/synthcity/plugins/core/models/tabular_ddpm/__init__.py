# mypy: allow-untyped-defs, allow-untyped-calls
# flake8: noqa: F401

# stdlib
from copy import deepcopy
from typing import Any, Optional, Union

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments
from torch import nn

# synthcity absolute
from synthcity.metrics.weighted_metrics import WeightedMetrics
from synthcity.utils.constants import DEVICE
from synthcity.utils.dataframe import discrete_columns

# synthcity relative
from .gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from .utils import TensorDataLoader


class TabDDPM(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_iter: int = 1000,
        lr: float = 0.002,
        weight_decay: float = 1e-4,
        batch_size: int = 1024,
        num_timesteps: int = 1000,
        gaussian_loss_type: str = "mse",
        scheduler: str = "cosine",
        device: Any = DEVICE,
        verbose: int = 0,
        log_interval: int = 10,
        print_interval: int = 100,
        # model params
        model_type: str = "mlp",
        rtdl_params: Optional[dict] = None,  # {'d_layers', 'dropout'}
        dim_label_emb: int = 128,
        # early stopping
        n_iter_min: int = 100,
        n_iter_print: int = 50,
        patience: int = 5,
    ) -> None:
        super().__init__()
        self.__dict__.update(locals())
        del self.self

    def _anneal_lr(self, epoch):
        frac_done = epoch / self.n_iter
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

    def fit(self, X: pd.DataFrame, cond: Any = None, **kwargs: Any) -> "TabDDPM":
        if cond is not None:
            n_labels = cond.nunique()
        else:
            n_labels = 0

        cat_cols = discrete_columns(X, return_counts=True)

        if cat_cols:
            ini_cols = X.columns
            cat_cols, cat_counts = zip(*cat_cols)
            # reorder the columns so that the categorical ones go to the end
            X = X[np.hstack([X.columns[~X.keys().isin(cat_cols)], cat_cols])]
            cur_cols = X.columns
            # find the permutation from the reordered columns to the original ones
            self._col_perm = np.argsort(cur_cols)[np.argsort(np.argsort(ini_cols))]
        else:
            cat_counts = [0]
            self._col_perm = np.arange(X.shape[1])

        model_params = dict(
            num_classes=n_labels,
            is_y_cond=cond is not None,
            rtdl_params=self.rtdl_params,
            dim_t=self.dim_label_emb,
        )

        tensors = [
            torch.tensor(X.values, dtype=torch.float32, device=self.device),
            np.repeat(None, len(X))
            if cond is None
            else torch.tensor(cond.values, dtype=torch.long, device=self.device),
        ]
        self.dataloader = TensorDataLoader(*tensors, batch_size=self.batch_size)

        self.diffusion = GaussianMultinomialDiffusion(
            model_type=self.model_type,
            model_params=model_params,
            num_categorical_features=cat_counts,
            num_numerical_features=X.shape[1] - len(cat_cols),
            gaussian_loss_type=self.gaussian_loss_type,
            num_timesteps=self.num_timesteps,
            scheduler=self.scheduler,
            device=self.device,
            verbose=self.verbose,
        ).to(self.device)

        self.ema_model = deepcopy(self.diffusion.denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.optimizer = torch.optim.AdamW(
            self.diffusion.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.loss_history = pd.DataFrame(columns=["step", "mloss", "gloss", "loss"])

        # if self.verbose:
        #     print("Starting training")
        #     print(self)

        steps = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_count = 0

        for epoch in range(self.n_iter):
            self.diffusion.train()

            for x, y in self.dataloader:
                self.optimizer.zero_grad()
                loss_multi, loss_gauss = self.diffusion.mixed_loss(x, y)
                loss = loss_multi + loss_gauss
                loss.backward()
                self.optimizer.step()

                self._anneal_lr(epoch + 1)

                curr_count += len(x)
                curr_loss_multi += loss_multi.item() * len(x)
                curr_loss_gauss += loss_gauss.item() * len(x)

                steps += 1
                if steps % self.log_interval == 0:
                    mloss = np.around(curr_loss_multi / curr_count, 4)
                    gloss = np.around(curr_loss_gauss / curr_count, 4)
                    if self.verbose and steps % self.print_interval == 0:
                        print(
                            f"Step {steps}: MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}"
                        )
                    self.loss_history.loc[len(self.loss_history)] = [
                        steps,
                        mloss,
                        gloss,
                        mloss + gloss,
                    ]
                    curr_count = 0
                    curr_loss_gauss = 0.0
                    curr_loss_multi = 0.0

                self._update_ema(
                    self.ema_model.parameters(), self.diffusion.parameters()
                )

        return self

    def generate(self, count: int, cond: Any = None) -> np.ndarray:
        self.diffusion.eval()
        if cond is not None:
            cond = torch.tensor(cond, dtype=torch.long, device=self.device)
        sample = self.diffusion.sample_all(count, cond).detach().cpu().numpy()
        sample = sample[:, self._col_perm]
        if self.verbose:
            print("Generated sample")
            print(sample)
        return sample
