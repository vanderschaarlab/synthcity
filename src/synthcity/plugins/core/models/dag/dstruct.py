# stdlib
from typing import Callable, Iterable, List, Tuple, Union

# third party
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping

# synthcity absolute
import synthcity.logger as log
import synthcity.plugins.core.models.dag.utils as ut
from synthcity.plugins.core.models.dag.data import BetaP, CustomDataModule
from synthcity.plugins.core.models.dag.dsl import NotearsMLP, NotearsSobolev

accelerator = "gpu" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NOTEARS(nn.Module):
    def __init__(
        self,
        dim: int,  # Dims of system
        nonlinear_dims: list = [10, 10, 1],  # Dims for non-linear arch
        sem_type: str = "mlp",
        rho: float = 1.0,  # NOTEARS parameters
        alpha: float = 1.0,
        lambda1: float = 0.0,
        lambda2: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.notears: Union[NotearsMLP, NotearsSobolev] = (
            NotearsMLP(dims=[dim, *nonlinear_dims]).to(DEVICE).double()
            if sem_type == "mlp"
            else NotearsSobolev(dim, 5).to(DEVICE).double()
        )

        self.rho = rho
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def _squared_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        return 0.5 / n * torch.sum((x_hat - x) ** 2)

    def h_func(self) -> torch.Tensor:
        return self.notears.h_func()

    def loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        x = x.double()
        loss = self._squared_loss(x, x_hat)
        h_val = self.notears.h_func()
        penalty = 0.5 * self.rho * h_val * h_val + self.alpha * h_val
        l2_reg = 0.5 * self.lambda2 * self.notears.l2_reg()
        l1_reg = self.lambda1 * self.notears.fc1_l1_reg()

        return loss + penalty + l2_reg + l1_reg

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_hat = self.notears(x)
        loss = self.loss(x, x_hat)

        return x_hat, loss


class DStruct(pl.LightningModule):
    def __init__(
        self,
        dim: int,
        dsl: Callable,
        dsl_config: dict,
        K: int = 3,
        lr: float = 0.001,
        lmbda: int = 2,
        s: int = 9,
        dag_type: str = "ER",
        h_tol: float = 1e-8,
        rho_max: float = 1e16,
        w_threshold: float = 0.3,
    ) -> None:

        super().__init__()

        self.h_tol, self.rho_max, self.w_threshold = h_tol, rho_max, w_threshold

        self.lr = lr
        self.K = K
        self.dim = dim
        self.lmbda = lmbda
        self.s = s

        self.automatic_optimization = False
        self.dsl_list = nn.ModuleList(
            [NOTEARS(dim=self.dim).to(DEVICE).double() for i in range(self.K)]
        )

        for i, dsl in enumerate(self.dsl_list):
            self.dsl_list[i].h = np.inf

        self.p = BetaP(K)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        (X,) = batch
        subsets = self.p(X)

        opts = self.optimizers()
        opt = opts[0]
        opt.zero_grad()

        if self.current_epoch < 0:
            return

        hs, rhos, alphas = [], [], []
        for i, dsl in enumerate(self.dsl_list):
            subset = subsets[i]

            alpha, rho, h = self._dual_ascent_step(subset, opts[i + 1], dsl)
            dsl.h = h

            hs.append(h)
            rhos.append(rho)
            alphas.append(alpha)

        h, rho, alpha = np.max(hs), np.min(rhos), np.mean(alphas)

        self.log_dict({"h": h, "rho": rho, "alpha": alpha})

    def _dual_ascent_step(
        self, x: torch.Tensor, optimizer: torch.optim.Optimizer, dsl: NOTEARS
    ) -> Tuple[float, float, float]:
        x = x.double()
        h_new = 0

        while dsl.rho < self.rho_max:

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                mse_loss = self._loss()
                self.log(
                    "mse_loss",
                    mse_loss.item(),
                    on_step=True,
                    logger=True,
                    prog_bar=True,
                )
                _, loss = dsl(x)

                self.log(
                    "dsl_loss", loss.item(), on_step=True, logger=True, prog_bar=True
                )
                loss += self.lmbda * mse_loss.item()
                self.log(
                    "total_loss", loss.item(), on_step=True, logger=True, prog_bar=True
                )

                self.manual_backward(loss)
                return loss

            optimizer.step(closure)

            with torch.no_grad():
                h_new = dsl.h_func().item()
            if h_new > 0.25 * dsl.h:
                dsl.rho *= 10
            else:
                break

        dsl.alpha += dsl.rho * h_new

        return dsl.alpha, dsl.rho, h_new

    def configure_optimizers(self) -> Iterable[torch.optim.Optimizer]:
        dsl_optimizers = [ut.LBFGSBScipy(dsl.parameters()) for dsl in self.dsl_list]

        self_optim = ut.LBFGSBScipy(self.dsl_list.parameters())

        return tuple([self_optim, *dsl_optimizers])

    def forward(self, threshold: float = 0.5, grad: bool = True) -> tuple:
        if grad:
            As = tuple([dsl.notears.fc1_to_adj_grad() for dsl in self.dsl_list])
            _As = torch.stack(As).mean(dim=0)
        else:
            As = np.array([dsl.notears.fc1_to_adj() for dsl in self.dsl_list])

            _As = np.mean(As, axis=0)

            _As[np.abs(_As) > threshold] = 1
            _As[np.abs(_As) <= threshold] = 0

        return As, _As

    def _loss(self) -> torch.Tensor:
        As, A_comp = self.forward()

        mask = torch.ones(A_comp.shape).to(DEVICE).double()
        mask.diagonal().zero_()

        A_comp.detach()
        A_comp.diagonal().zero_()

        lloss = 0
        mse = nn.MSELoss()
        for A_est in As:
            lloss += mse(A_est * mask, A_comp)
        return lloss

    def get_dag(self, threshold: float = 0.5) -> torch.Tensor:
        _, A = self.forward(threshold=threshold, grad=False)

        return A


def get_dstruct_dag(
    X: pd.DataFrame,
    K: int = 3,  # amount of subsets for D-Struct
    n_iter: int = 100,
    lmbda: int = 1,
    batch_size: int = 256,
    seed: int = 0,
    nt_h_tol: float = 1e-8,  # minimum value for NOTEARS
    nt_rho_max: float = 1e18,  # maximum value for NATEARGS
    compress: bool = True,
) -> List:
    pl.seed_everything(seed)
    n, dim = X.shape
    dsl = NOTEARS
    dsl_config = {"dim": dim, "sem_type": "sob"}

    s = int(dim * (dim - 1) / 2 - 1)
    Dataset = CustomDataModule(X, batch_size=batch_size)
    Dataset.setup(stage="fit")

    model = (
        DStruct(
            dim=dim,
            dsl=dsl,
            dsl_config=dsl_config,
            K=K,
            lmbda=lmbda,
            h_tol=nt_h_tol,
            rho_max=nt_rho_max,
            s=s,
        )
        .to(DEVICE)
        .double()
    )
    trainer = pl.Trainer(
        accelerator=accelerator,
        log_every_n_steps=1,
        max_epochs=n_iter,
        callbacks=[
            EarlyStopping(monitor="h", stopping_threshold=nt_h_tol),
            EarlyStopping(monitor="rho", stopping_threshold=nt_rho_max),
        ],
    )
    trainer.fit(model, datamodule=Dataset)

    model.eval()

    for threshold in np.linspace(start=0, stop=1, num=100):
        dag = model.get_dag(threshold=threshold)
        if ut.is_dag(dag):
            log.info(f"Is DAG for {threshold}")
            break

    if not compress:
        return dag

    out = []

    for row in range(dim):
        for col in range(dim):
            if dag[row][col] == 1:
                out.append((row, col))

    return out
