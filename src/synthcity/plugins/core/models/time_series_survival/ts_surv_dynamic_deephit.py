# stdlib
from copy import deepcopy
from typing import Any, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from auton_survival.models.dsm import DeepRecurrentSurvivalMachines
from auton_survival.models.dsm.utilities import _get_padded_features, get_optimizer
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.models.mlp import MLP
from synthcity.utils.constants import DEVICE
from synthcity.utils.reproducibility import enable_reproducible_results

# synthcity relative
from ._base import TimeSeriesSurvivalPlugin


class DynamicDeephitTimeSeriesSurvival(TimeSeriesSurvivalPlugin):
    def __init__(
        self,
        n_iter: int = 1000,
        batch_size: int = 100,
        lr: float = 1e-3,
        n_layers_hidden: int = 1,
        n_units_hidden: int = 40,
        split: int = 100,
        rnn_type: str = "GRU",
        alpha: float = 0.34,
        beta: float = 0.27,
        sigma: float = 0.21,
        seed: int = 0,
        dropout: float = 0.06,
        device: Any = DEVICE,
        patience: int = 20,
        **kwargs: Any
    ) -> None:
        super().__init__()
        enable_reproducible_results(seed)

        self.lr = lr
        self.batch_size = batch_size
        self.n_iter = n_iter

        self.model = DynamicDeepHitModel(
            split=split,
            layers_rnn=n_layers_hidden,
            hidden_rnn=n_units_hidden,
            rnn_type=rnn_type,
            alpha=alpha,
            beta=beta,
            sigma=sigma,
            dropout=dropout,
            patience=patience,
        )

    def _merge_data(
        self,
        static: Optional[np.ndarray],
        temporal: np.ndarray,
    ) -> np.ndarray:
        if static is None:
            return temporal

        merged = []
        for idx, item in enumerate(temporal):
            local_static = static[idx].reshape(1, -1)
            local_static = np.repeat(local_static, len(temporal[idx]), axis=0)
            tst = np.concatenate([temporal[idx], local_static], axis=1)
            merged.append(tst)

        return np.array(merged, dtype=object)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        static: Optional[np.ndarray],
        temporal: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
    ) -> TimeSeriesSurvivalPlugin:
        data = self._merge_data(static, temporal)

        self.model.fit(
            data,
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
        data = self._merge_data(static, temporal)

        return pd.DataFrame(
            self.model.predict_risk(data, time_horizons), columns=time_horizons
        )

    @staticmethod
    def name() -> str:
        return "dynamic_deephit"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="n_units_hidden", low=10, high=100, step=10),
            IntegerDistribution(name="n_layers_hidden", low=1, high=4),
            CategoricalDistribution(name="batch_size", choices=[100, 200, 500]),
            CategoricalDistribution(name="lr", choices=[1e-2, 1e-3, 1e-4]),
            CategoricalDistribution(name="rnn_type", choices=["LSTM", "GRU", "RNN"]),
            FloatDistribution(name="alpha", low=0.0, high=0.5),
            FloatDistribution(name="sigma", low=0.0, high=0.5),
            FloatDistribution(name="beta", low=0.0, high=0.5),
            FloatDistribution(name="dropout", low=0.0, high=0.2),
        ]


class DynamicDeepHitModel(DeepRecurrentSurvivalMachines):
    """
    This implementation considers that the last event happen at the same time for each patient
    The CIF is therefore simplified

    Args:
            DeepRecurrentSurvivalMachines
    """

    def __init__(
        self,
        split: int = 100,
        layers_rnn: int = 2,
        hidden_rnn: int = 100,
        rnn_type: str = "LSTM",
        dropout: float = 0.1,
        alpha: float = 0.1,
        beta: float = 0.1,
        sigma: float = 0.1,
        patience: int = 20,
        device: Any = DEVICE,
    ) -> None:

        self.split = split
        self.split_time = None

        self.layers_rnn = layers_rnn
        self.hidden_rnn = hidden_rnn
        self.rnn_type = rnn_type

        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

        self.device = device
        self.dropout = dropout

        self.patience = patience

        self.torch_model: Optional[DynamicDeepHitTorch] = None

    def _gen_torch_model(self, inputdim: int, optimizer: Any, risks: Any) -> Any:
        model = (
            DynamicDeepHitTorch(
                inputdim,
                self.split,
                self.layers_rnn,
                self.hidden_rnn,
                rnn_type=self.rnn_type,
                optimizer=optimizer,
                dropout=self.dropout,
                risks=risks,
                device=self.device,
            )
            .float()
            .to(self.device)
        )
        return model

    def fit(
        self,
        x: np.ndarray,
        t: np.ndarray,
        e: np.ndarray,
        vsize: float = 0.15,
        val_data: Optional[Tuple] = None,
        iters: int = 1,
        learning_rate: float = 1e-3,
        batch_size: int = 100,
        optimizer: str = "Adam",
        random_state: int = 100,
    ) -> Any:
        discretized_t, self.split_time = self.discretize(t, self.split, self.split_time)
        processed_data = self._preprocess_training_data(
            x, discretized_t, e, vsize, val_data, random_state
        )
        x_train, t_train, e_train, x_val, t_val, e_val = processed_data
        inputdim = x_train.shape[-1]

        maxrisk = int(np.nanmax(e_train.cpu().numpy()))

        model = self._gen_torch_model(inputdim, optimizer, risks=maxrisk)
        model = train_ddh(
            model,
            x_train,
            t_train,
            e_train,
            x_val,
            t_val,
            e_val,
            self.alpha,
            self.beta,
            self.sigma,
            n_iter=iters,
            lr=learning_rate,
            bs=batch_size,
            device=self.device,
            train_patience=self.patience,
        )

        self.torch_model = model.eval()

        return self

    def discretize(
        self, t: np.ndarray, split: int, split_time: Optional[int] = None
    ) -> Tuple:
        """
        Discretize the survival horizon

        Args:
                t (List of Array): Time of events
                split (int): Number of bins
                split_time (List, optional): List of bins (must be same length than split). Defaults to None.

        Returns:
                List of Array: Disretized events time
        """
        if split_time is None:
            _, split_time = np.histogram(t, split - 1)
        t_discretized = np.array(
            [np.digitize(t_, split_time, right=True) - 1 for t_ in t], dtype=object
        )
        return t_discretized, split_time

    def _preprocess_test_data(self, x: np.ndarray) -> torch.Tensor:
        data = torch.from_numpy(_get_padded_features(x)).float()
        return data

    def _preprocess_training_data(
        self,
        x: np.ndarray,
        t: np.ndarray,
        e: np.ndarray,
        vsize: float,
        val_data: Optional[Tuple],
        random_state: int,
    ) -> Tuple:
        """RNNs require different preprocessing for variable length sequences"""

        idx = list(range(x.shape[0]))
        np.random.seed(random_state)
        np.random.shuffle(idx)

        x = _get_padded_features(x)
        x_train, t_train, e_train = x[idx], t[idx], e[idx]

        x_train = torch.from_numpy(x_train).float().to(self.device)
        t_train = (
            torch.from_numpy(np.asarray(t_train).astype(float)).float().to(self.device)
        )
        e_train = (
            torch.from_numpy(np.asarray(e_train).astype(int)).float().to(self.device)
        )

        if val_data is None:

            vsize = int(vsize * x_train.shape[0])

            x_val, t_val, e_val = x_train[-vsize:], t_train[-vsize:], e_train[-vsize:]

            x_train = x_train[:-vsize]
            t_train = t_train[:-vsize]
            e_train = e_train[:-vsize]

        else:

            x_val, t_val, e_val = val_data

            x_val = _get_padded_features(x_val)
            t_val, _ = self.discretize(t_val, self.split, self.split_time)

            x_val = torch.from_numpy(x_val).float()
            t_val = torch.from_numpy(np.asarray(t_val).astype(float)).float()
            e_val = torch.from_numpy(np.asarray(e_val).astype(int)).float()

        return (x_train, t_train, e_train, x_val, t_val, e_val)

    def compute_nll(self, x: torch.Tensor, t: torch.Tensor, e: torch.Tensor) -> float:
        if self.torch_model is None:
            raise Exception(
                "The model has not been fitted yet. Please fit the "
                + "model using the `fit` method on some training data "
                + "before calling `_eval_nll`."
            )
        discretized_t, _ = self.discretize(t, self.split, self.split_time)
        processed_data = self._preprocess_training_data(x, discretized_t, e, 0, None, 0)
        _, _, _, x_val, t_val, e_val = processed_data
        return total_loss(self.torch_model, x_val, t_val, e_val, 0, 1, 1).item()

    def predict_survival(
        self,
        x: np.ndarray,
        t: np.ndarray,
        risk: int = 1,
        all_step: bool = False,
        bs: int = 100,
    ) -> np.ndarray:
        if self.torch_model is None:
            raise Exception(
                "The model has not been fitted yet. Please fit the "
                + "model using the `fit` method on some training data "
                + "before calling `predict_survival`."
            )
        lens = [len(x_) for x_ in x]

        if all_step:
            new_x = []
            for x_, l_ in zip(x, lens):
                new_x += [x_[: li + 1] for li in range(l_)]
            x = new_x

        if not isinstance(t, list):
            t = [t]
        t = self.discretize([t], self.split, self.split_time)[0][0]
        x = self._preprocess_test_data(x)
        batches = int(len(x) / bs) + 1
        scores: dict = {t_: [] for t_ in t}
        for j in range(batches):
            xb = x[j * bs : (j + 1) * bs]
            _, f = self.torch_model(xb)
            for t_ in t:
                scores[t_].extend(
                    torch.cumsum(f[int(risk) - 1], dim=1)[:, t_]
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

        output = []
        for t_ in t:
            output.append(scores[t_])

        return 1 - np.asarray(output).T

    def predict_risk(self, x: np.ndarray, t: np.ndarray, **args: Any) -> np.ndarray:
        if self.torch_model is not None:
            return 1 - self.predict_survival(x, t, **args)

        raise Exception(
            "The model has not been fitted yet. Please fit the "
            + "model using the `fit` method on some training data "
            + "before calling `predict_risk`."
        )


class DynamicDeepHitTorch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers_rnn: int,
        hidden_rnn: int,
        rnn_type: str = "LSTM",
        optimizer: str = "Adam",
        dropout: float = 0.1,
        risks: int = 1,
        device: Any = DEVICE,
    ) -> None:
        super(DynamicDeepHitTorch, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.optimizer = optimizer
        self.risks = risks
        self.rnn_type = rnn_type
        self.device = device
        self.dropout = dropout

        # RNN model for longitudinal data
        if self.rnn_type == "LSTM":
            self.embedding = nn.LSTM(
                input_dim, hidden_rnn, layers_rnn, bias=False, batch_first=True
            )
        if self.rnn_type == "RNN":
            self.embedding = nn.RNN(
                input_dim,
                hidden_rnn,
                layers_rnn,
                bias=False,
                batch_first=True,
                nonlinearity="relu",
            )
        if self.rnn_type == "GRU":
            self.embedding = nn.GRU(
                input_dim, hidden_rnn, layers_rnn, bias=False, batch_first=True
            )

        # Longitudinal network
        self.longitudinal = MLP(
            task_type="regression",
            n_units_in=hidden_rnn,
            n_units_out=input_dim,
            n_layers_hidden=layers_rnn,
            n_units_hidden=hidden_rnn,
            dropout=self.dropout,
        )

        # Attention mechanism
        self.attention = MLP(
            task_type="regression",
            n_units_in=input_dim + hidden_rnn,
            n_units_out=1,
            dropout=self.dropout,
            n_layers_hidden=layers_rnn,
            n_units_hidden=hidden_rnn,
        )
        self.attention_soft = nn.Softmax(1)  # On temporal dimension

        # Cause specific network
        self.cause_specific = []
        for r in range(self.risks):
            self.cause_specific.append(
                MLP(
                    task_type="regression",
                    n_units_in=input_dim + hidden_rnn,
                    n_units_out=output_dim,
                    dropout=self.dropout,
                    n_layers_hidden=layers_rnn,
                    n_units_hidden=hidden_rnn,
                )
            )
        self.cause_specific = nn.ModuleList(self.cause_specific)

        # Probability
        self.soft = nn.Softmax(dim=-1)  # On all observed output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward function that is called when data is passed through DynamicDeepHit.
        """
        # RNN representation - Nan values for not observed data
        x = x.clone()
        inputmask = torch.isnan(x[:, :, 0])
        x[inputmask] = -1

        hidden, _ = self.embedding(x)

        # Longitudinal modelling
        longitudinal_prediction = self.longitudinal(hidden)

        # Attention using last observation to predict weight of all previously observed
        # Extract last observation (the one used for predictions)
        last_observations = (~inputmask).sum(axis=1) - 1
        last_observations_idx = last_observations.unsqueeze(1).repeat(1, x.size(1))
        index = torch.arange(x.size(1)).repeat(x.size(0), 1).to(self.device)

        last = index == last_observations_idx
        x_last = x[last]

        # Concatenate all previous with new to measure attention
        concatenation = torch.cat(
            [hidden, x_last.unsqueeze(1).repeat(1, x.size(1), 1)], -1
        )

        # Compute attention and normalize
        attention = self.attention(concatenation).squeeze(-1)
        attention[
            index >= last_observations_idx
        ] = -1e10  # Want soft max to be zero as values not observed
        attention[last_observations > 0] = self.attention_soft(
            attention[last_observations > 0]
        )  # Weight previous observation
        attention[last_observations == 0] = 0  # No context for only one observation

        # Risk networks
        # The original paper is not clear on how the last observation is
        # combined with the temporal sum, other code was concatenating them
        outcomes = []
        attention = attention.unsqueeze(2).repeat(1, 1, hidden.size(2))
        hidden_attentive = torch.sum(attention * hidden, axis=1)
        hidden_attentive = torch.cat([hidden_attentive, x_last], 1)
        for cs_nn in self.cause_specific:
            outcomes.append(cs_nn(hidden_attentive))

        # Soft max for probability distribution
        outcomes_t = torch.cat(outcomes, dim=1)
        outcomes_t = self.soft(outcomes_t)

        outcomes = [
            outcomes_t[:, i * self.output_dim : (i + 1) * self.output_dim]
            for i in range(self.risks)
        ]
        return longitudinal_prediction, outcomes


def train_ddh(
    model: Any,
    x_train: torch.Tensor,
    t_train: torch.Tensor,
    e_train: torch.Tensor,
    x_valid: torch.Tensor,
    t_valid: torch.Tensor,
    e_valid: torch.Tensor,
    alpha: float,
    beta: float,
    sigma: float,
    n_iter: int = 10000,
    lr: float = 1e-3,
    bs: int = 100,
    vbs: int = 500,
    train_patience: int = 20,
    device: Any = DEVICE,
) -> Any:

    optimizer = get_optimizer(model, lr)

    patience, old_loss = 0, np.inf
    nbatches = int(x_train.shape[0] / bs) + 1
    valbatches = int(x_valid.shape[0] / vbs) + 1

    for i in range(n_iter):
        model.train()
        for j in range(nbatches):
            xb = x_train[j * bs : (j + 1) * bs]
            tb = t_train[j * bs : (j + 1) * bs]
            eb = e_train[j * bs : (j + 1) * bs]

            if xb.shape[0] == 0:
                continue

            optimizer.zero_grad()
            loss = total_loss(model, xb, tb, eb, alpha, beta, sigma, device=device)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss: torch.Tensor = 0
        for j in range(valbatches):
            xb = x_valid[j * bs : (j + 1) * bs]
            tb = t_valid[j * bs : (j + 1) * bs]
            eb = e_valid[j * bs : (j + 1) * bs]

            if xb.shape[0] == 0:
                continue

            valid_loss += total_loss(model, xb, tb, eb, alpha, beta, sigma)

        valid_loss = valid_loss.item()
        if valid_loss < old_loss:
            patience = 0
            old_loss = valid_loss
            best_param = deepcopy(model.state_dict())
        else:
            if patience == train_patience:
                break
            else:
                patience += 1

    model.load_state_dict(best_param)
    return model


def negative_log_likelihood(
    outcomes: list,
    cif: torch.Tensor,
    t: torch.Tensor,
    e: torch.Tensor,
    device: Any = DEVICE,
) -> torch.Tensor:
    """
    Compute the log likelihood loss
    This function is used to compute the survival loss
    """
    loss, censored_cif = 0, 0
    for k, ok in enumerate(outcomes):
        # Censored cif
        censored_cif += cif[k][e == 0][:, t[e == 0]]

        # Uncensored
        selection = e == (k + 1)
        loss += torch.sum(torch.log(ok[selection][:, t[selection]] + 1e-10))

    # Censored loss
    loss += torch.sum(torch.log(1 - censored_cif + 1e-10))
    return -loss / len(outcomes)


def ranking_loss(
    cif: torch.Tensor,
    t: torch.Tensor,
    e: torch.Tensor,
    sigma: float,
    device: Any = DEVICE,
) -> torch.Tensor:
    """
    Penalize wrong ordering of probability
    Equivalent to a C Index
    This function is used to penalize wrong ordering in the survival prediction
    """
    loss = 0
    # Data ordered by time
    for k, cifk in enumerate(cif):
        for ci, ti in zip(cifk[e - 1 == k], t[e - 1 == k]):
            # For all events: all patients that didn't experience event before
            # must have a lower risk for that cause
            if torch.sum(t > ti) > 0:
                # TODO: When data are sorted in time -> wan we make it even faster ?
                loss += torch.mean(torch.exp((cifk[t > ti][:, ti] - ci[ti])) / sigma)

    return loss / len(cif)


def longitudinal_loss(
    longitudinal_prediction: torch.Tensor, x: torch.Tensor, device: Any = DEVICE
) -> torch.Tensor:
    """
    Penalize error in the longitudinal predictions
    This function is used to compute the error made by the RNN

    NB: In the paper, they seem to use different losses for continuous and categorical
    But this was not reflected in the code associated (therefore we compute MSE for all)

    NB: Original paper mentions possibility of different alphas for each risk
    But take same for all (for ranking loss)
    """
    length = (~torch.isnan(x[:, :, 0])).sum(axis=1) - 1

    # Create a grid of the column index
    index = torch.arange(x.size(1)).repeat(x.size(0), 1).to(device)

    # Select all predictions until the last observed
    prediction_mask = index <= (length - 1).unsqueeze(1).repeat(1, x.size(1))

    # Select all observations that can be predicted
    observation_mask = index <= length.unsqueeze(1).repeat(1, x.size(1))
    observation_mask[:, 0] = False  # Remove first observation

    return torch.nn.MSELoss(reduction="mean")(
        longitudinal_prediction[prediction_mask], x[observation_mask]
    )


def total_loss(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    e: torch.Tensor,
    alpha: float,
    beta: float,
    sigma: float,
    device: Any = DEVICE,
) -> torch.Tensor:
    longitudinal_prediction, outcomes = model(x.float())
    t, e = t.long(), e.int()

    # Compute cumulative function from prediced outcomes
    cif = [torch.cumsum(ok, 1) for ok in outcomes]

    return (
        (1 - alpha - beta)
        * longitudinal_loss(longitudinal_prediction, x, device=device)
        + alpha * ranking_loss(cif, t, e, sigma, device=device)
        + beta * negative_log_likelihood(outcomes, cif, t, e, device=device)
    )
