# stdlib
from typing import Any, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments
from sklearn.utils import shuffle
from torch import nn

# synthcity absolute
import synthcity.logger as log
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.models.mlp import MLP
from synthcity.plugins.core.models.time_series_survival.utils import (
    BreslowEstimator,
    get_padded_features,
)
from synthcity.plugins.core.models.transformer import TransformerModel
from synthcity.utils.constants import DEVICE
from synthcity.utils.reproducibility import enable_reproducible_results

# synthcity relative
from ._base import TimeSeriesSurvivalPlugin


class DeepCoxPHTimeSeriesSurvival(TimeSeriesSurvivalPlugin):
    def __init__(
        self,
        n_iter: int = 1000,
        batch_size: int = 100,
        lr: float = 1e-3,
        n_layers_hidden: int = 4,
        n_units_hidden: int = 50,
        random_state: int = 0,
        dropout: float = 0.17,
        patience: int = 20,
        rnn_type: str = "Transformer",
        device: Any = DEVICE,
    ) -> None:
        super().__init__()
        enable_reproducible_results(random_state)

        self.lr = lr
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_layers_hidden = n_layers_hidden
        self.n_units_hidden = n_units_hidden
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.patience = patience
        self.random_state = random_state
        self.device = device

    def _merge_data(
        self,
        static: Optional[np.ndarray],
        temporal: np.ndarray,
        observation_times: np.ndarray,
    ) -> np.ndarray:
        if static is None:
            static = np.zeros((len(temporal), 0))

        merged = []
        for idx, item in enumerate(temporal):
            local_static = static[idx].reshape(1, -1)
            local_static = np.repeat(local_static, len(temporal[idx]), axis=0)
            tst = np.concatenate(
                [temporal[idx], local_static, observation_times[idx].reshape(-1, 1)],
                axis=1,
            )
            merged.append(tst)

        return np.array(merged, dtype=object)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        static: Optional[np.ndarray],
        temporal: np.ndarray,
        observation_times: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
    ) -> TimeSeriesSurvivalPlugin:
        data = self._merge_data(static, temporal, observation_times)

        self.model = DeepRecurrentCoxPH(
            data[0].shape[-1],
            n_layers_hidden=self.n_layers_hidden,
            n_units_hidden=self.n_units_hidden,
            batch_size=self.batch_size,
            lr=self.lr,
            n_iter=self.n_iter,
            rnn_type=self.rnn_type,
            random_state=self.random_state,
            patience=self.patience,
            dropout=self.dropout,
            device=self.device,
        )

        self.model.fit(
            data,
            T,
            E,
        )
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

        data = self._merge_data(static, temporal, observation_times)

        raw = self.model.predict_risk(data, time_horizons)
        out = []

        offset = -1
        for item in temporal:
            offset += len(item)
            out.append(raw[offset])

        if len(raw) != offset + 1:
            raise RuntimeError(f"Invalid prediction offset {len(raw)} {offset}")

        return pd.DataFrame(out, columns=time_horizons)

    @staticmethod
    def name() -> str:
        return "deep_recurrent_coxph"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="n_units_hidden", low=10, high=100, step=10),
            IntegerDistribution(name="n_layers_hidden", low=1, high=4),
            CategoricalDistribution(name="batch_size", choices=[100, 200, 500]),
            CategoricalDistribution(name="lr", choices=[1e-2, 1e-3, 1e-4]),
            CategoricalDistribution(
                name="rnn_type", choices=["LSTM", "GRU", "RNN", "Transformer"]
            ),
            FloatDistribution(name="dropout", low=0.0, high=0.2),
        ]


class DeepRecurrentCoxPH(nn.Module):
    def __init__(
        self,
        n_units_in: int,
        rnn_type: str = "LSTM",
        n_layers_hidden: int = 2,
        n_units_hidden: int = 200,
        optimizer: str = "Adam",
        n_iter: int = 1000,
        n_iter_print: int = 10,
        val_size: float = 0.1,
        random_state: int = 0,
        batch_size: int = 100,
        lr: float = 1e-3,
        patience: int = 10,
        dropout: float = 0.1,
        device: Any = DEVICE,
    ) -> None:
        super().__init__()

        self.rnn_type = rnn_type
        self.optimizer = optimizer
        self.random_state = random_state
        self.n_iter = n_iter
        self.n_iter_print = n_iter_print
        self.val_size = val_size
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience

        self.n_units_hidden = n_units_hidden
        self.n_layers_hidden = n_layers_hidden

        self.expert = MLP(
            task_type="regression",
            n_units_in=n_units_hidden,
            n_units_out=1,
            dropout=dropout,
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            device=device,
        )

        rnn_params = {
            "input_size": n_units_in,
            "hidden_size": n_units_hidden,
            "num_layers": n_layers_hidden,
            "dropout": 0 if n_layers_hidden == 1 else dropout,
            "batch_first": True,
            "bias": False,
        }
        if self.rnn_type == "LSTM":
            self.embedding = nn.LSTM(**rnn_params)
        elif self.rnn_type == "RNN":
            self.embedding = nn.RNN(**rnn_params)
        elif self.rnn_type == "GRU":
            self.embedding = nn.GRU(**rnn_params)
        elif self.rnn_type == "Transformer":
            self.embedding = TransformerModel(
                n_units_in,
                n_units_hidden,
                n_layers_hidden=n_layers_hidden,
                dropout=dropout,
            )
        else:
            raise RuntimeError(f"unknown rnn type {self.rnn_type}")

        self.embedding.to(device)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().clone().to(self.device)
        inputmask = ~torch.isnan(x[:, :, 0]).reshape(-1)
        x[torch.isnan(x)] = -1

        if self.rnn_type in ["LSTM", "RNN", "GRU"]:
            xrep, _ = self.embedding(x)
        else:
            xrep = self.embedding(x)

        xrep = xrep.contiguous().view(-1, self.n_units_hidden)
        xrep = xrep[inputmask]
        xrep = nn.ReLU6()(xrep)

        dim = xrep.shape[0]

        return self.expert(xrep.view(dim, -1))

    def fit(
        self,
        x: np.ndarray,
        t: np.ndarray,
        e: np.ndarray,
    ) -> "DeepRecurrentCoxPH":
        self.train()
        processed_data = self._preprocess_training_data(x, t, e)

        x_train, t_train, e_train, x_val, t_val, e_val = processed_data

        t_train_ = self._reshape_tensor_with_nans(t_train)
        e_train_ = self._reshape_tensor_with_nans(e_train)
        t_val_ = self._reshape_tensor_with_nans(t_val)
        e_val_ = self._reshape_tensor_with_nans(e_val)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        valc = np.inf
        patience_ = 0

        losses = []

        for epoch in range(self.n_iter):
            _ = self.train_step(x_train, t_train, e_train, optimizer)
            valcn = self.test_step(x_val, t_val_, e_val_)

            losses.append(valcn)

            if epoch % self.n_iter_print == 0:
                log.debug(f"[Epoch: epoch] loss: {valcn}")

            if valcn > valc:
                patience_ += 1
            else:
                valc = valcn
                patience_ = 0

            if patience_ == self.patience:
                break

        self.breslow_spline = self.fit_breslow(x_train, t_train_, e_train_)

        self.eval()
        self.fitted = True

        return self

    def predict_risk(self, x: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
        return 1 - self.predict_survival(x, t)

    def predict_survival(
        self, x: np.ndarray, t: Optional[np.ndarray] = None
    ) -> np.ndarray:
        r"""Returns the estimated survival probability at time \( t \),
          \( \widehat{\mathbb{P}}(T > t|X) \) for some input data \( x \).
        Parameters
        ----------
        x: np.ndarray
            A numpy array of the input features, \( x \).
        t: list or float
            a list or float of the times at which survival probability is
            to be computed
        Returns:
          np.array: numpy array of the survival probabilites at each time in t.
        """
        if not self.fitted:
            raise Exception(
                "The model has not been fitted yet. Please fit the "
                + "model using the `fit` method on some training data "
                + "before calling `predict_survival`."
            )

        x = self._preprocess_test_data(x)

        if t is not None:
            if not isinstance(t, list):
                t = [t]

        lrisks = self(x).detach().cpu().numpy()

        unique_times = self.breslow_spline.baseline_survival_.x

        raw_predictions = self.breslow_spline.get_survival_function(lrisks)
        raw_predictions = np.array([pred.y for pred in raw_predictions])

        predictions = pd.DataFrame(data=raw_predictions, columns=unique_times)

        if t is None:
            return predictions
        else:
            return self.__interpolate_missing_times(predictions.T, t)

    def _preprocess_test_data(self, x: np.ndarray) -> torch.Tensor:
        if isinstance(x, pd.DataFrame):
            x = x.values

        return torch.from_numpy(get_padded_features(x)).float()

    def _preprocess_training_data(
        self,
        x: np.ndarray,
        t: np.ndarray,
        e: np.ndarray,
    ) -> Tuple:
        """RNNs require different preprocessing for variable length sequences"""

        idx = list(range(x.shape[0]))
        np.random.shuffle(idx)

        x = get_padded_features(x)

        x_train, t_train, e_train = x[idx], t[idx], e[idx]

        x_train = torch.from_numpy(x_train).float()
        t_train = torch.from_numpy(t_train.astype(float)).float()
        e_train = torch.from_numpy(e_train.astype(int)).float()

        vsize = int(self.val_size * x_train.shape[0])

        x_val, t_val, e_val = x_train[-vsize:], t_train[-vsize:], e_train[-vsize:]

        x_train = x_train[:-vsize]
        t_train = t_train[:-vsize]
        e_train = e_train[:-vsize]

        return (x_train, t_train, e_train, x_val, t_val, e_val)

    def _reshape_tensor_with_nans(self, data: torch.Tensor) -> torch.Tensor:
        """Helper function to unroll padded RNN inputs."""
        data = data.reshape(-1)
        return data[~torch.isnan(data)]

    def partial_ll_loss(
        self,
        xb: torch.Tensor,
        tb: torch.Tensor,
        eb: torch.Tensor,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        lrisks = self(xb)

        tb = tb + eps * np.random.random(len(tb))
        sindex = np.argsort(-tb)

        tb = tb[sindex]
        eb = eb[sindex]

        lrisks = lrisks[sindex]
        lrisksdenom = torch.logcumsumexp(lrisks, dim=0)

        plls = lrisks - lrisksdenom
        pll = plls[eb == 1]

        pll = torch.sum(pll)

        return -pll

    def fit_breslow(
        self, x: torch.Tensor, t: torch.Tensor, e: torch.Tensor
    ) -> BreslowEstimator:
        return BreslowEstimator().fit(
            self(x).detach().cpu().numpy(), e.numpy(), t.numpy()
        )

    def train_step(
        self, x: torch.Tensor, t: torch.Tensor, e: torch.Tensor, optimizer: Any
    ) -> float:

        x, t, e = shuffle(x, t, e, random_state=self.random_state)

        n = x.shape[0]

        batches = (n // self.batch_size) + 1

        epoch_loss = 0.0

        for i in range(batches):

            xb = x[i * self.batch_size : (i + 1) * self.batch_size]
            tb = t[i * self.batch_size : (i + 1) * self.batch_size]
            eb = e[i * self.batch_size : (i + 1) * self.batch_size]

            # Training Step
            torch.enable_grad()
            optimizer.zero_grad()
            loss = self.partial_ll_loss(
                xb,
                self._reshape_tensor_with_nans(tb),
                self._reshape_tensor_with_nans(eb),
            )
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss)

        return epoch_loss / n

    def test_step(
        self, x: torch.Tensor, t: torch.Tensor, e: torch.Tensor
    ) -> torch.Tensor:

        with torch.no_grad():
            loss = float(self.partial_ll_loss(x, t, e))

        return loss / x.shape[0]

    def __interpolate_missing_times(
        self, survival_predictions: np.ndarray, times: list
    ) -> np.ndarray:

        nans = np.full(survival_predictions.shape[1], np.nan)
        not_in_index = list(set(times) - set(survival_predictions.index))

        for idx in not_in_index:
            survival_predictions.loc[idx] = nans
        return (
            survival_predictions.sort_index(axis=0)
            .interpolate(method="bfill")
            .T[times]
            .values
        )
