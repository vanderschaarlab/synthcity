# stdlib
from collections.abc import Iterator
from copy import deepcopy
from typing import Any, Optional, Sequence

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

# synthcity absolute
from synthcity.logger import info
from synthcity.metrics.weighted_metrics import WeightedMetrics
from synthcity.utils.callbacks import Callback, ValidationMixin
from synthcity.utils.constants import DEVICE
from synthcity.utils.dataframe import discrete_columns

# synthcity relative
from .gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion


class TabDDPM(nn.Module, ValidationMixin):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_iter: int = 1000,
        lr: float = 0.002,
        weight_decay: float = 0.0001,
        batch_size: int = 1024,
        num_timesteps: int = 1000,
        is_classification: bool = False,
        gaussian_loss_type: str = "mse",
        scheduler: str = "cosine",
        callbacks: Sequence[Callback] = (),
        device: torch.device = DEVICE,
        log_interval: int = 10,
        # validation
        valid_size: float = 0,
        valid_metric: Optional[WeightedMetrics] = None,
        # model params
        model_type: str = "mlp",
        model_params: Optional[dict] = None,
        dim_embed: int = 128,
        # early stopping
        n_iter_min: int = 100,
        patience: int = 5,
    ) -> None:
        nn.Module.__init__(self)
        ValidationMixin.__init__(
            self,
            valid_metric=valid_metric,  # type: ignore
            valid_size=valid_size,
            callbacks=callbacks,
        )
        self.__dict__.update(locals())
        del self.self

    def _anneal_lr(self, epoch: int) -> None:
        frac_done = epoch / self.n_iter
        lr = self.lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _update_ema(
        self,
        target_params: Iterator,
        source_params: Iterator,
        rate: float = 0.999,
    ) -> None:
        """
        Update target parameters to be closer to those of source parameters using
        an exponential moving average.
        :param target_params: the target parameter sequence.
        :param source_params: the source parameter sequence.
        :param rate: the EMA rate (closer to 1 means slower).
        """
        for targ, src in zip(target_params, source_params):
            targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)

    def fit(
        self, X: pd.DataFrame, cond: Optional[pd.Series] = None, **kwargs: Any
    ) -> "TabDDPM":

        X = self._set_val_data(X)

        self.on_fit_begin()

        if self.is_classification and cond is not None:
            if np.ndim(cond) != 1:
                raise ValueError("cond must be a 1D array")
            self.n_classes = cond.nunique()
        else:
            self.n_classes = 0

        self.feature_names = X.columns
        cat_cols = discrete_columns(X, return_counts=True)

        if cat_cols:
            cat_cols, cat_counts = zip(*cat_cols)
            num_cols = X.columns.difference(cat_cols)
            # reorder the columns so that the categorical ones go to the end
            X = X[list(num_cols) + list(cat_cols)]
            self.feature_names_out = X.columns
        else:
            cat_counts = [0]
            self.feature_names_out = self.feature_names

        dataset = TensorDataset(
            torch.tensor(X.values, dtype=torch.float32, device=self.device),
            torch.tensor([torch.nan] * len(X), dtype=torch.float32, device=self.device)
            if cond is None
            else torch.tensor(
                cond.values,
                dtype=torch.long if self.is_classification else torch.float32,
                device=self.device,
            ),
        )

        self.dataloader = DataLoader(dataset, batch_size=self.batch_size)

        self.diffusion = GaussianMultinomialDiffusion(
            model_type=self.model_type,
            model_params=self.model_params,
            num_categorical_features=cat_counts,
            num_numerical_features=X.shape[1] - len(cat_cols),
            gaussian_loss_type=self.gaussian_loss_type,
            num_timesteps=self.num_timesteps,
            num_classes=self.n_classes,
            conditional=cond is not None,
            dim_emb=self.dim_embed,
            scheduler=self.scheduler,
            device=self.device,
        ).to(self.device)

        self.ema_model = deepcopy(self.diffusion.denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.optimizer = torch.optim.AdamW(
            self.diffusion.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.loss_history = []
        self.val_history = []

        steps = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_count = 0
        pbar = trange(self.n_iter, desc="Epoch", leave=True)

        for epoch in pbar:
            self.train()
            self.on_epoch_begin()

            for x, y in self.dataloader:
                self.optimizer.zero_grad()
                args = (x,) if cond is None else (x, y)
                loss_multi, loss_gauss = self.diffusion.mixed_loss(*args)
                loss = loss_multi + loss_gauss
                loss.backward()
                self.optimizer.step()

                self._anneal_lr(epoch + 1)

                curr_count += len(x)
                curr_loss_multi += loss_multi.item() * len(x)
                curr_loss_gauss += loss_gauss.item() * len(x)
                steps += 1

                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                loss = mloss + gloss

                self._update_ema(
                    self.ema_model.parameters(), self.diffusion.parameters()
                )

                if steps % self.log_interval == 0:
                    self.loss_history.append([steps, mloss, gloss, loss])
                    curr_count = 0
                    curr_loss_gauss = 0.0
                    curr_loss_multi = 0.0

            self.eval()
            self.on_epoch_end()

            if self.valid_score is not None:
                self.val_history.append(self.valid_score)  # type: ignore
                pbar.set_postfix(loss=loss, val=self.valid_score)
            else:
                pbar.set_postfix(loss=loss)

            if self.should_stop:
                info(f"Early stopped at epoch {epoch}")
                break

        self.loss_history = pd.DataFrame(
            self.loss_history, columns=["step", "mloss", "gloss", "loss"]
        ).set_index("step")

        self.on_fit_end()

        return self

    def generate(self, count: int, cond: Any = None) -> pd.DataFrame:
        self.diffusion.eval()
        if cond is not None:
            cond = torch.tensor(cond, dtype=torch.long, device=self.device)
        sample = self.diffusion.sample_all(count, cond).detach().cpu().numpy()
        df = pd.DataFrame(sample, columns=self.feature_names_out)
        return df[self.feature_names]
