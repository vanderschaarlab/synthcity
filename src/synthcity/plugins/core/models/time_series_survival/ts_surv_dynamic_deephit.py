# stdlib
from copy import deepcopy
from typing import Any, List, Optional, Tuple, Union

# third party
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.models.mlp import MLP
from synthcity.plugins.core.models.time_series_survival.utils import get_padded_features
from synthcity.plugins.core.models.transformer import TransformerModel
from synthcity.plugins.core.models.ts_model import TimeSeriesLayer
from synthcity.utils.constants import DEVICE
from synthcity.utils.reproducibility import enable_reproducible_results

# synthcity relative
from ._base import TimeSeriesSurvivalPlugin

rnn_modes = ["GRU", "LSTM", "RNN", "Transformer"]
output_modes = [
    "MLP",
    "Transformer",
    "LSTM",
    "GRU",
    "RNN",
    "TCN",
    "InceptionTime",
    "InceptionTimePlus",
    "ResCNN",
    "XCM",
]


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
        random_state: int = 0,
        dropout: float = 0.06,
        device: Any = DEVICE,
        patience: int = 20,
        output_type: str = "MLP",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        enable_reproducible_results(random_state)
        if rnn_type not in rnn_modes:
            raise ValueError(f"Supported modes: {rnn_modes}")
        if output_type not in output_modes:
            raise ValueError(f"Supported output modes: {output_modes}")

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
            lr=lr,
            batch_size=batch_size,
            n_iter=n_iter,
            output_type=output_type,
            device=device,
        )

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
                [
                    temporal[idx],
                    local_static,
                    np.asarray(observation_times[idx]).reshape(-1, 1),
                ],
                axis=1,
            )
            merged.append(tst)

        return np.array(merged, dtype=object)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        static: Optional[np.ndarray],
        temporal: Union[np.ndarray, List],
        observation_times: Union[np.ndarray, List],
        T: Union[np.ndarray, List],
        E: Union[np.ndarray, List],
    ) -> TimeSeriesSurvivalPlugin:
        static = np.asarray(static)
        temporal = np.asarray(temporal)
        observation_times = np.asarray(observation_times)
        T = np.asarray(T)
        E = np.asarray(E)

        data = self._merge_data(static, temporal, observation_times)

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
        temporal: Union[np.ndarray, List],
        observation_times: Union[np.ndarray, List],
        time_horizons: List,
    ) -> np.ndarray:
        "Predict risk"
        static = np.asarray(static)
        temporal = np.asarray(temporal)
        observation_times = np.asarray(observation_times)

        data = self._merge_data(static, temporal, observation_times)

        return pd.DataFrame(
            self.model.predict_risk(data, time_horizons), columns=time_horizons
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict_emb(
        self,
        static: Optional[np.ndarray],
        temporal: Union[np.ndarray, List],
        observation_times: Union[np.ndarray, List],
    ) -> np.ndarray:
        "Predict embeddings"
        static = np.asarray(static)
        temporal = np.asarray(temporal)
        observation_times = np.asarray(observation_times)

        data = self._merge_data(static, temporal, observation_times)

        return self.model.predict_emb(data).detach().cpu().numpy()

    @staticmethod
    def name() -> str:
        return "dynamic_deephit"

    @staticmethod
    def hyperparameter_space(
        *args: Any, prefix: str = "", **kwargs: Any
    ) -> List[Distribution]:
        return [
            IntegerDistribution(
                name=f"{prefix}n_units_hidden", low=10, high=100, step=10
            ),
            IntegerDistribution(name=f"{prefix}n_layers_hidden", low=1, high=4),
            CategoricalDistribution(
                name=f"{prefix}batch_size", choices=[100, 200, 500]
            ),
            CategoricalDistribution(name=f"{prefix}lr", choices=[1e-2, 1e-3, 1e-4]),
            CategoricalDistribution(name=f"{prefix}rnn_type", choices=rnn_modes),
            CategoricalDistribution(name=f"{prefix}output_type", choices=output_modes),
            FloatDistribution(name=f"{prefix}alpha", low=0.0, high=0.5),
            FloatDistribution(name=f"{prefix}sigma", low=0.0, high=0.5),
            FloatDistribution(name=f"{prefix}beta", low=0.0, high=0.5),
            FloatDistribution(name=f"{prefix}dropout", low=0.0, high=0.2),
        ]


class DynamicDeepHitModel:
    """
    This implementation considers that the last event happen at the same time for each patient
    The CIF is therefore simplified
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
        lr: float = 1e-3,
        batch_size: int = 100,
        n_iter: int = 1000,
        device: Any = DEVICE,
        val_size: float = 0.1,
        random_state: int = 0,
        clipping_value: int = 1,
        output_type: str = "MLP",
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
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.val_size = val_size
        self.clipping_value = clipping_value

        self.patience = patience
        self.random_state = random_state
        self.output_type = output_type

        self.model: Optional[DynamicDeepHitLayers] = None

    def _setup_model(
        self, inputdim: int, seqlen: int, risks: int
    ) -> "DynamicDeepHitLayers":
        return (
            DynamicDeepHitLayers(
                inputdim,
                seqlen,
                self.split,
                self.layers_rnn,
                self.hidden_rnn,
                rnn_type=self.rnn_type,
                dropout=self.dropout,
                risks=risks,
                device=self.device,
                output_type=self.output_type,
            )
            .float()
            .to(self.device)
        )

    def fit(
        self,
        x: np.ndarray,
        t: np.ndarray,
        e: np.ndarray,
    ) -> Any:
        discretized_t, self.split_time = self.discretize(t, self.split, self.split_time)
        processed_data = self._preprocess_training_data(x, discretized_t, e)
        x_train, t_train, e_train, x_val, t_val, e_val = processed_data
        inputdim = x_train.shape[-1]
        seqlen = x_train.shape[-2]

        maxrisk = int(np.nanmax(e_train.cpu().numpy()))

        self.model = self._setup_model(inputdim, seqlen, risks=maxrisk)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        patience, old_loss = 0, np.inf
        nbatches = int(x_train.shape[0] / self.batch_size) + 1
        valbatches = int(x_val.shape[0] / self.batch_size) + 1

        for i in range(self.n_iter):
            self.model.train()
            for j in range(nbatches):
                xb = x_train[j * self.batch_size : (j + 1) * self.batch_size]
                tb = t_train[j * self.batch_size : (j + 1) * self.batch_size]
                eb = e_train[j * self.batch_size : (j + 1) * self.batch_size]

                if xb.shape[0] == 0:
                    continue

                optimizer.zero_grad()
                loss = self.total_loss(xb, tb, eb)
                loss.backward()

                if self.clipping_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clipping_value
                    )

                optimizer.step()

            self.model.eval()
            valid_loss: torch.Tensor = 0
            for j in range(valbatches):
                xb = x_val[j * self.batch_size : (j + 1) * self.batch_size]
                tb = t_val[j * self.batch_size : (j + 1) * self.batch_size]
                eb = e_val[j * self.batch_size : (j + 1) * self.batch_size]

                if xb.shape[0] == 0:
                    continue

                valid_loss += self.total_loss(xb, tb, eb)

            if torch.isnan(valid_loss):
                raise RuntimeError("NaNs detected in the total loss")

            valid_loss = valid_loss.item()

            if valid_loss < old_loss:
                patience = 0
                old_loss = valid_loss
                best_param = deepcopy(self.model.state_dict())
            else:
                patience += 1

            if patience == self.patience:
                break

        self.model.load_state_dict(best_param)
        self.model.eval()

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
        data = (
            torch.from_numpy(get_padded_features(x, pad_size=self.pad_size))
            .float()
            .to(self.device)
        )
        return data

    def _preprocess_training_data(
        self,
        x: np.ndarray,
        t: np.ndarray,
        e: np.ndarray,
    ) -> Tuple:
        """RNNs require different preprocessing for variable length sequences"""

        idx = list(range(x.shape[0]))
        np.random.seed(self.random_state)
        np.random.shuffle(idx)

        x = get_padded_features(x)
        self.pad_size = x.shape[1]
        x_train, t_train, e_train = x[idx], t[idx], e[idx]

        x_train = torch.from_numpy(x_train.astype(float)).float().to(self.device)
        t_train = torch.from_numpy(t_train.astype(float)).float().to(self.device)
        e_train = torch.from_numpy(e_train.astype(int)).float().to(self.device)

        vsize = int(self.val_size * x_train.shape[0])

        x_val, t_val, e_val = x_train[-vsize:], t_train[-vsize:], e_train[-vsize:]

        x_train = x_train[:-vsize]
        t_train = t_train[:-vsize]
        e_train = e_train[:-vsize]

        return (x_train, t_train, e_train, x_val, t_val, e_val)

    def predict_emb(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        if self.model is None:
            raise Exception(
                "The model has not been fitted yet. Please fit the "
                + "model using the `fit` method on some training data "
                + "before calling `predict_survival`."
            )
        x = self._preprocess_test_data(x)

        _, emb = self.model.forward_emb(x)

        return emb

    def predict_survival(
        self,
        x: np.ndarray,
        t: np.ndarray,
        risk: int = 1,
        all_step: bool = False,
        bs: int = 100,
    ) -> np.ndarray:
        if self.model is None:
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

        t = self.discretize([t], self.split, self.split_time)[0][0]
        x = self._preprocess_test_data(x)
        batches = int(len(x) / bs) + 1
        scores: dict = {t_: [] for t_ in t}
        for j in range(batches):
            xb = x[j * self.batch_size : (j + 1) * self.batch_size]
            _, f = self.model(xb)
            for t_ in t:
                pred = (
                    torch.cumsum(f[int(risk) - 1], dim=1)[:, t_]
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

                if isinstance(pred, list):
                    scores[t_].extend(pred)
                else:
                    scores[t_].append(pred)

        output = []
        for t_ in t:
            output.append(scores[t_])

        return 1 - np.asarray(output).T

    def predict_risk(self, x: np.ndarray, t: np.ndarray, **args: Any) -> np.ndarray:
        return 1 - self.predict_survival(x, t, **args)

    def negative_log_likelihood(
        self,
        outcomes: torch.Tensor,
        cif: torch.Tensor,
        t: torch.Tensor,
        e: torch.Tensor,
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
        loss += torch.sum(torch.log(nn.ReLU()(1 - censored_cif) + 1e-10))
        return -loss / len(outcomes)

    def ranking_loss(
        self,
        cif: torch.Tensor,
        t: torch.Tensor,
        e: torch.Tensor,
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
                    loss += torch.mean(
                        torch.exp((cifk[t > ti][:, ti] - ci[ti])) / self.sigma
                    )

        return loss / len(cif)

    def longitudinal_loss(
        self, longitudinal_prediction: torch.Tensor, x: torch.Tensor
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
        index = torch.arange(x.size(1)).repeat(x.size(0), 1).to(self.device)

        # Select all predictions until the last observed
        prediction_mask = index <= (length - 1).unsqueeze(1).repeat(1, x.size(1))

        # Select all observations that can be predicted
        observation_mask = index <= length.unsqueeze(1).repeat(1, x.size(1))
        observation_mask[:, 0] = False  # Remove first observation

        return torch.nn.MSELoss(reduction="mean")(
            longitudinal_prediction[prediction_mask], x[observation_mask]
        )

    def total_loss(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        e: torch.Tensor,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Invalid model for loss")

        longitudinal_prediction, outcomes = self.model(x.float())
        if torch.isnan(longitudinal_prediction).sum() != 0:
            raise RuntimeError("NaNs detected in the longitudinal_prediction")

        t, e = t.long(), e.int()

        # Compute cumulative function from prediced outcomes
        cif = [torch.cumsum(ok, 1) for ok in outcomes]

        return (
            (1 - self.alpha - self.beta)
            * self.longitudinal_loss(longitudinal_prediction, x)
            + self.alpha * self.ranking_loss(cif, t, e)
            + self.beta * self.negative_log_likelihood(outcomes, cif, t, e)
        )


class DynamicDeepHitLayers(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        output_dim: int,
        layers_rnn: int,
        hidden_rnn: int,
        rnn_type: str = "LSTM",
        dropout: float = 0.1,
        risks: int = 1,
        output_type: str = "MLP",
        device: Any = DEVICE,
    ) -> None:
        super(DynamicDeepHitLayers, self).__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.risks = risks
        self.rnn_type = rnn_type
        self.device = device
        self.dropout = dropout

        # RNN model for longitudinal data
        if self.rnn_type == "LSTM":
            self.embedding = nn.LSTM(
                input_dim, hidden_rnn, layers_rnn, bias=False, batch_first=True
            )
        elif self.rnn_type == "RNN":
            self.embedding = nn.RNN(
                input_dim,
                hidden_rnn,
                layers_rnn,
                bias=False,
                batch_first=True,
                nonlinearity="relu",
            )
        elif self.rnn_type == "GRU":
            self.embedding = nn.GRU(
                input_dim, hidden_rnn, layers_rnn, bias=False, batch_first=True
            )
        elif self.rnn_type == "Transformer":
            self.embedding = TransformerModel(
                input_dim, hidden_rnn, n_layers_hidden=layers_rnn, dropout=dropout
            )
        else:
            raise RuntimeError(f"Unknown rnn_type {rnn_type}")
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
        if output_type == "MLP":
            self.attention = MLP(
                task_type="regression",
                n_units_in=input_dim + hidden_rnn,
                n_units_out=1,
                dropout=self.dropout,
                n_layers_hidden=layers_rnn,
                n_units_hidden=hidden_rnn,
            )
        else:
            self.attention = TimeSeriesLayer(
                n_static_units_in=0,
                n_temporal_units_in=input_dim + hidden_rnn,
                n_temporal_window=seq_len,
                n_units_out=seq_len,
                n_temporal_units_hidden=hidden_rnn,
                n_temporal_layers_hidden=layers_rnn,
                mode=output_type,
                dropout=self.dropout,
                device=device,
            )
        self.attention_soft = nn.Softmax(1)  # On temporal dimension
        self.output_type = output_type

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

    def forward_attention(
        self, x: torch.Tensor, inputmask: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
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
        if self.output_type == "MLP":
            attention = self.attention(concatenation).squeeze(-1)
        else:
            attention = self.attention(
                torch.zeros(len(concatenation), 0).to(self.device), concatenation
            ).squeeze(-1)
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
        attention = attention.unsqueeze(2).repeat(1, 1, hidden.size(2))
        hidden_attentive = torch.sum(attention * hidden, axis=1)
        return torch.cat([hidden_attentive, x_last], 1)

    def forward_emb(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward function that is called when data is passed through DynamicDeepHit.
        """
        # RNN representation - Nan values for not observed data
        x = x.clone()
        inputmask = torch.isnan(x[:, :, 0])
        x[torch.isnan(x)] = -1

        if torch.isnan(x).sum() != 0:
            raise RuntimeError("NaNs detected in the input")

        if self.rnn_type in ["GRU", "LSTM", "RNN"]:
            hidden, _ = self.embedding(x)
        else:
            hidden = self.embedding(x)

        if torch.isnan(hidden).sum() != 0:
            raise RuntimeError("NaNs detected in the embeddings")

        # Longitudinal modelling
        longitudinal_prediction = self.longitudinal(hidden)
        if torch.isnan(longitudinal_prediction).sum() != 0:
            raise RuntimeError("NaNs detected in the longitudinal_prediction")

        hidden_attentive = self.forward_attention(x, inputmask, hidden)

        return longitudinal_prediction, hidden_attentive

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward function that is called when data is passed through DynamicDeepHit.
        """
        # RNN representation - Nan values for not observed data
        longitudinal_prediction, hidden_attentive = self.forward_emb(x)

        outcomes = []
        for cs_nn in self.cause_specific:
            outcomes.append(cs_nn(hidden_attentive))

        # Soft max for probability distribution
        outcomes_t = torch.cat(outcomes, dim=1)
        outcomes_t = self.soft(outcomes_t)
        if torch.isnan(outcomes_t).sum() != 0:
            raise RuntimeError("NaNs detected in the outcome")

        outcomes = [
            outcomes_t[:, i * self.output_dim : (i + 1) * self.output_dim]
            for i in range(self.risks)
        ]
        return longitudinal_prediction, outcomes
