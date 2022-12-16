"""
PyTorch implementation for "Adversarial Time-to-Event Modeling"
Paper: https://arxiv.org/pdf/1804.03184.pdf
"""
# stdlib
from typing import Any, Callable, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# synthcity absolute
import synthcity.logger as log
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.models.mlp import MLP
from synthcity.plugins.core.models.time_to_event.metrics import (
    c_index,
    expected_time_error,
)
from synthcity.utils.constants import DEVICE
from synthcity.utils.reproducibility import enable_reproducible_results

# synthcity relative
from ._base import TimeToEventPlugin


class TimeEventGAN(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_features: int,
        n_units_latent: int,
        model_search_n_iter: Optional[int] = None,
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 250,
        generator_nonlin: str = "leaky_relu",
        generator_nonlin_out: Optional[List[Tuple[str, int]]] = None,
        generator_n_iter: int = 1000,
        generator_batch_norm: bool = False,
        generator_dropout: float = 0,
        generator_loss: Optional[Callable] = None,
        generator_lr: float = 2e-4,
        generator_weight_decay: float = 1e-3,
        generator_residual: bool = True,
        generator_opt_betas: tuple = (0.9, 0.999),
        discriminator_n_layers_hidden: int = 3,
        discriminator_n_units_hidden: int = 300,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_batch_norm: bool = False,
        discriminator_dropout: float = 0.1,
        discriminator_loss: Optional[Callable] = None,
        discriminator_lr: float = 2e-4,
        discriminator_weight_decay: float = 1e-3,
        discriminator_opt_betas: tuple = (0.9, 0.999),
        patience: int = 10,
        batch_size: int = 100,
        n_iter_print: int = 50,
        random_state: int = 0,
        n_iter_min: int = 100,
        clipping_value: int = 0,
        device: Any = DEVICE,
    ) -> None:
        super(TimeEventGAN, self).__init__()

        self.device = device
        self.n_features = n_features
        self.n_units_latent = n_units_latent

        if model_search_n_iter is not None:
            generator_n_iter = model_search_n_iter

        self.generator = MLP(
            task_type="regression",
            n_units_in=n_features + n_units_latent,
            n_units_out=1,  # time to event
            n_layers_hidden=generator_n_layers_hidden,
            n_units_hidden=generator_n_units_hidden,
            nonlin=generator_nonlin,
            nonlin_out=generator_nonlin_out,
            n_iter=generator_n_iter,
            batch_norm=generator_batch_norm,
            dropout=generator_dropout,
            loss=generator_loss,
            random_state=random_state,
            lr=generator_lr,
            residual=generator_residual,
            opt_betas=generator_opt_betas,
        ).to(self.device)

        self.discriminator = MLP(
            task_type="regression",
            n_units_in=n_features + 1,
            n_units_out=1,  # fake/true
            n_layers_hidden=discriminator_n_layers_hidden,
            n_units_hidden=discriminator_n_units_hidden,
            nonlin=discriminator_nonlin,
            nonlin_out=[("none", 1)],
            n_iter=discriminator_n_iter,
            batch_norm=discriminator_batch_norm,
            dropout=discriminator_dropout,
            loss=discriminator_loss,
            random_state=random_state,
            lr=discriminator_lr,
            opt_betas=discriminator_opt_betas,
        ).to(self.device)

        # training
        self.generator_n_iter = generator_n_iter
        self.discriminator_n_iter = discriminator_n_iter
        self.n_iter_print = n_iter_print
        self.n_iter_min = n_iter_min
        self.batch_size = batch_size
        self.clipping_value = clipping_value
        self.patience = patience

        self.random_state = random_state
        enable_reproducible_results(random_state)

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
    ) -> "TimeEventGAN":
        Xt = self._check_tensor(X)
        Tt = self._check_tensor(T)
        Et = self._check_tensor(E)

        self._train(
            Xt,
            Tt,
            Et,
        )

        return self

    def generate(self, X: np.ndarray) -> np.ndarray:
        X = self._check_tensor(X).float()

        return self(X).cpu().numpy()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self.generator.eval()

        fixed_noise = torch.randn(len(X), self.n_units_latent, device=self.device)
        with torch.no_grad():
            return self.generator(torch.hstack([X, fixed_noise])).detach().cpu()

    def dataloader(
        self, X: torch.Tensor, T: torch.Tensor, E: torch.Tensor
    ) -> Tuple[DataLoader, TensorDataset]:
        X_train, X_val, T_train, T_val, E_train, E_val = train_test_split(
            X.cpu(),
            T.cpu(),
            E.cpu(),
            stratify=E.cpu(),
            random_state=self.random_state,
        )
        train_dataset = TensorDataset(
            self._check_tensor(X_train),
            self._check_tensor(T_train),
            self._check_tensor(E_train),
        )
        val_dataset = TensorDataset(
            self._check_tensor(X_val),
            self._check_tensor(T_val),
            self._check_tensor(E_val),
        )

        return DataLoader(
            train_dataset, batch_size=self.batch_size, pin_memory=False
        ), DataLoader(val_dataset, batch_size=len(val_dataset), pin_memory=False)

    def _generate_training_outputs(
        self, X: torch.Tensor, T: torch.Tensor, E: torch.Tensor
    ) -> tuple:
        # Train with non-censored true batch
        Xnc = X[E == 1].to(self.device)
        Tnc = T[E == 1].to(self.device)
        true_features_time = torch.hstack([Xnc, Tnc.reshape(-1, 1)])

        true_output = self.discriminator(true_features_time).squeeze().float()

        # Train with fake batch
        noise = torch.randn(len(X), self.n_units_latent, device=self.device)
        noise = torch.hstack([X, noise])
        fake_T = self.generator(noise)

        fake_features_time = torch.hstack([X, fake_T.reshape(-1, 1)])

        fake_output = self.discriminator(fake_features_time.detach()).squeeze()

        return true_output, fake_output

    def _train_epoch_generator(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        E: torch.Tensor,
    ) -> float:
        # Update the G network
        self.generator.optimizer.zero_grad()

        # Evaluate noncensored error
        Xnc = X[E == 1].to(self.device)
        Tnc = T[E == 1].to(self.device)

        batch_size = len(Xnc)

        noise = torch.randn(batch_size, self.n_units_latent, device=self.device)
        noncen_input = torch.hstack([Xnc, noise])
        fake_T = self.generator(noncen_input).squeeze()

        errG_noncen = nn.MSELoss()(
            fake_T, Tnc
        )  # fake_T should be == T for noncensored data

        # Evaluate censored error
        Xc = X[E == 0].to(self.device)
        Tc = T[E == 0].to(self.device)

        batch_size = len(Xc)

        noise = torch.randn(batch_size, self.n_units_latent, device=self.device)
        cen_input = torch.hstack([Xc, noise])
        fake_T = self.generator(cen_input)

        errG_cen = torch.mean(
            nn.ReLU()(Tc - fake_T)
        )  # fake_T should be >= T for censored data

        # Discriminator loss
        real_output, fake_output = self._generate_training_outputs(X, T, E)

        errG_discr = 0
        if real_output.dim() > 0 and len(real_output) > 0:
            errG_discr += torch.mean(real_output)

        if fake_output.dim() > 0 and len(fake_output) > 0:
            errG_discr -= torch.mean(fake_output)

        # Calculate G's loss based on this output
        errG = errG_noncen + errG_cen + errG_discr

        if errG.isnan().sum() != 0:
            raise RuntimeError("NaNs detected in the generator loss")

        # Calculate gradients for G
        errG.backward()

        # Update G
        if self.clipping_value > 0:
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), self.clipping_value
            )
        self.generator.optimizer.step()

        # Return loss
        return errG.item()

    def _train_epoch_discriminator(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        E: torch.Tensor,
    ) -> float:
        # Update the D network
        errors = []

        for epoch in range(self.discriminator_n_iter):
            self.discriminator.zero_grad()

            (
                true_output,
                fake_output,
            ) = self._generate_training_outputs(X, T, E)

            act = nn.Sigmoid()
            errD: torch.Tensor = 0
            if true_output.dim() > 0 and len(true_output) > 0:
                errD -= torch.mean(torch.log(act(true_output)))

            if fake_output.dim() > 0 and len(fake_output) > 0:
                errD -= torch.mean(torch.log(1 - act(fake_output)))

            if errD.isnan().sum() != 0:
                raise RuntimeError("NaNs detected in the discriminator loss")
            errD.backward()

            # Update D
            if self.clipping_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), self.clipping_value
                )
            self.discriminator.optimizer.step()

            errors.append(errD.item())

        return np.mean(errors)

    def _train_epoch(
        self,
        loader: DataLoader,
    ) -> Tuple[float, float]:

        G_losses = []
        D_losses = []

        for i, data in enumerate(loader):
            G_losses.append(
                self._train_epoch_generator(
                    *data,
                )
            )
            D_losses.append(
                self._train_epoch_discriminator(
                    *data,
                )
            )

        return np.mean(G_losses), np.mean(D_losses)

    def _train(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        E: torch.Tensor,
    ) -> "TimeEventGAN":
        X = self._check_tensor(X).float()
        T = self._check_tensor(T).float()
        E = self._check_tensor(E).long()

        # Load Dataset
        loader, val_loader = self.dataloader(X, T, E)

        best_exp_tte_err = 9999
        patience = 0

        # Train loop
        for i in range(self.generator_n_iter):
            g_loss, d_loss = self._train_epoch(loader)
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i + 1) % self.n_iter_print == 0:
                with torch.no_grad():
                    Xval, Tval, Eval = next(iter(val_loader))
                    Xdf = Xval.cpu().numpy()
                    Tdf = pd.Series(Tval.cpu().numpy())
                    Edf = pd.Series(Eval.cpu().numpy(), index=Tdf.index)

                    Tpred = pd.Series(
                        self.generate(Xdf).squeeze(),
                        index=Tdf.index,
                    )

                    c_index_val = c_index(Tdf, Edf, Tpred)
                    exp_err_val = expected_time_error(Tdf, Edf, Tpred)

                    if best_exp_tte_err > exp_err_val:
                        patience = 0
                        best_exp_tte_err = exp_err_val
                    else:
                        patience += 1

                    if patience > self.patience:
                        log.debug(
                            f"No improvement for {patience} iterations. stopping..."
                        )
                        break

                log.debug(
                    f"[{i}/{self.generator_n_iter}]\tLoss_D: {d_loss}\tLoss_G: {g_loss}\tC-Index val: {c_index_val} Expected time err: {exp_err_val}"
                )

        return self

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)


class DATETimeToEvent(TimeToEventPlugin):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

        self.kwargs = kwargs

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> "TimeToEventPlugin":
        "Training logic"
        self._fit_censoring_model(X, T, Y)

        self.model = TimeEventGAN(
            n_features=X.shape[1], n_units_latent=X.shape[1], **self.kwargs
        )

        self.scaler_X = MinMaxScaler()
        enc_X = self.scaler_X.fit_transform(X)

        self.scaler_T = MinMaxScaler()
        enc_T = self.scaler_T.fit_transform(T.values.reshape(-1, 1)).squeeze()

        self.model.fit(enc_X, enc_T, Y)

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(self, X: pd.DataFrame) -> pd.Series:
        "Predict time-to-event"

        enc_X = self.scaler_X.fit_transform(X)

        enc_preds = self.model.generate(enc_X)
        preds = self.scaler_T.inverse_transform(enc_preds).squeeze()

        return pd.Series(preds, index=X.index)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict_any(self, X: pd.DataFrame, E: pd.Series) -> pd.Series:
        "Predict time-to-event"

        result = pd.Series([0] * len(X), index=E.index)

        if (E == 1).sum() > 0:
            result[E == 1] = self.predict(X[E == 1])
        if (E == 0).sum() > 0:
            result[E == 0] = self._predict_censoring(X[E == 0])

        return result

    @staticmethod
    def name() -> str:
        return "date"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="generator_n_layers_hidden", low=1, high=5),
            IntegerDistribution(
                name="generator_n_units_hidden", low=100, high=500, step=50
            ),
            CategoricalDistribution(
                name="generator_nonlin", choices=["relu", "leaky_relu"]
            ),
            IntegerDistribution(
                name="generator_n_iter", low=1000, high=3000, step=1000
            ),
            FloatDistribution(name="generator_dropout", low=0, high=0.2),
            CategoricalDistribution(name="generator_lr", choices=[1e-2, 1e-3, 1e-4]),
            IntegerDistribution(name="discriminator_n_layers_hidden", low=1, high=5),
            IntegerDistribution(
                name="discriminator_n_units_hidden", low=100, high=500, step=50
            ),
            CategoricalDistribution(
                name="discriminator_nonlin", choices=["relu", "leaky_relu"]
            ),
            FloatDistribution(name="discriminator_dropout", low=0, high=0.2),
            CategoricalDistribution(
                name="discriminator_lr", choices=[1e-2, 1e-3, 1e-4]
            ),
            CategoricalDistribution(name="batch_size", choices=[100, 250, 500, 1000]),
        ]
