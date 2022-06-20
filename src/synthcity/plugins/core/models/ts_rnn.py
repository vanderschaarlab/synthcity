# stdlib
from typing import Any, Callable, List, Optional, Tuple

# third party
import numpy as np
import torch
from pydantic import validate_arguments
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, sampler

# synthcity absolute
import synthcity.logger as log
from synthcity.plugins.core.models.mlp import MLP, MultiActivationHead, get_nonlin
from synthcity.utils.constants import DEVICE
from synthcity.utils.reproducibility import enable_reproducible_results


class WindowLinearLayer(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_static_units_in: int,
        n_temporal_units_in: int,
        window_size: int,
        n_units_out: int,
        n_units_hidden: int = 100,
        n_layers: int = 1,
        dropout: float = 0,
        nonlin: Optional[str] = "relu",
        device: Any = DEVICE,
    ) -> None:
        super(WindowLinearLayer, self).__init__()

        self.device = device
        self.window_size = window_size
        self.model = MLP(
            task_type="regression",
            n_units_in=n_static_units_in + n_temporal_units_in * window_size,
            n_units_out=n_units_out,
            n_layers_hidden=n_layers,
            n_units_hidden=n_units_hidden,
            dropout=dropout,
            nonlin=nonlin,
            device=device,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(
        self, static_data: torch.Tensor, temporal_data: torch.Tensor
    ) -> torch.Tensor:
        assert len(static_data) == len(temporal_data)
        batch_size, seq_len, n_feats = temporal_data.shape
        temporal_batch = temporal_data[:, seq_len - self.window_size :, :].reshape(
            batch_size, n_feats * self.window_size
        )
        batch = torch.cat([static_data, temporal_batch], axis=1)

        return self.model(batch).to(self.device)


class TimeSeriesRNN(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        task_type: str,  # regression, classification
        n_static_units_in: int,
        n_temporal_units_in: int,
        output_shape: List[int],
        n_static_units_hidden: int = 100,
        n_static_layers_hidden: int = 2,
        n_temporal_units_hidden: int = 100,
        n_temporal_layers_hidden: int = 2,
        n_iter: int = 500,
        mode: str = "RNN",  # RNN, LSTM, GRU
        n_iter_print: int = 10,
        batch_size: int = 150,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        window_size: int = 1,
        device: Any = DEVICE,
        dataloader_sampler: Optional[sampler.Sampler] = None,
        nonlin_out: Optional[List[Tuple[str, int]]] = None,
        loss: Optional[Callable] = None,
        dropout: float = 0,
        nonlin: Optional[str] = "relu",
        random_state: int = 0,
        clipping_value: int = 1,
    ) -> None:
        super(TimeSeriesRNN, self).__init__()

        enable_reproducible_results(random_state)

        assert task_type in ["classification", "regression"]
        assert len(output_shape) > 0

        self.task_type = task_type

        if loss is not None:
            self.loss = loss
        elif task_type == "regression":
            self.loss = nn.MSELoss()
        elif task_type == "classification":
            self.loss = nn.CrossEntropyLoss()

        self.n_iter = n_iter
        self.n_iter_print = n_iter_print
        self.batch_size = batch_size
        self.n_static_units_in = n_static_units_in
        self.n_temporal_units_in = n_temporal_units_in
        self.n_static_units_hidden = n_static_units_hidden
        self.n_temporal_units_hidden = n_temporal_units_hidden
        self.n_static_layers_hidden = n_static_layers_hidden
        self.n_temporal_layers_hidden = n_temporal_layers_hidden
        self.device = device
        self.window_size = window_size
        self.dataloader_sampler = dataloader_sampler
        self.lr = lr
        self.output_shape = output_shape
        self.n_units_out = np.prod(self.output_shape)
        self.clipping_value = clipping_value

        temporal_params = {
            "input_size": self.n_temporal_units_in + 1,
            "hidden_size": self.n_temporal_units_hidden,
            "num_layers": self.n_temporal_layers_hidden,
            "dropout": 0 if self.n_temporal_layers_hidden == 1 else dropout,
            "batch_first": True,
        }
        temporal_models = {
            "RNN": nn.RNN,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
        }

        self.temporal_layer = temporal_models[mode](**temporal_params).to(self.device)
        self.mode = mode

        self.out = WindowLinearLayer(
            n_static_units_in=n_static_units_in,
            n_temporal_units_in=self.n_temporal_units_hidden,
            window_size=self.window_size,
            n_units_out=self.n_units_out,
            n_layers=n_static_layers_hidden,
            dropout=dropout,
            nonlin=nonlin,
            device=device,
        )

        self.out_activation: Optional[nn.Module] = None
        self.n_act_out: Optional[int] = None

        if nonlin_out is not None:
            self.n_act_out = 0
            activations = []
            for nonlin, nonlin_len in nonlin_out:
                self.n_act_out += nonlin_len
                activations.append((get_nonlin(nonlin), nonlin_len))

            if self.n_units_out % self.n_act_out != 0:
                raise RuntimeError(
                    f"Shape mismatch for the output layer. Expected length {self.n_units_out}, but got {nonlin_out} with length {self.n_act_out}"
                )
            self.out_activation = MultiActivationHead(activations, device=device)
        elif self.task_type == "classification":
            self.n_act_out = self.n_units_out
            self.out_activation = MultiActivationHead(
                [(nn.Softmax(dim=-1), self.n_units_out)], device=device
            )

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )  # optimize all rnn parameters

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(
        self,
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        temporal_horizons: torch.Tensor,
    ) -> torch.Tensor:
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)

        assert torch.isnan(static_data).sum() == 0
        assert torch.isnan(temporal_data).sum() == 0
        assert torch.isnan(temporal_horizons).sum() == 0

        temporal_data_merged = torch.cat(
            [temporal_data, temporal_horizons.unsqueeze(2)], dim=2
        )
        assert torch.isnan(temporal_data_merged).sum() == 0

        X_interm, _ = self.temporal_layer(temporal_data_merged)
        assert torch.isnan(X_interm).sum() == 0

        # choose r_out at the last <window size> steps
        pred = self.out(static_data, X_interm)

        if self.out_activation is not None:
            pred = pred.reshape(-1, self.n_act_out)
            pred = self.out_activation(pred)

        pred = pred.reshape(-1, *self.output_shape)

        return pred

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(
        self,
        static_data: np.ndarray,
        temporal_data: np.ndarray,
        temporal_horizons: np.ndarray,
    ) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            temporal_data_t = self._check_tensor(temporal_data).float()
            temporal_horizons_t = self._check_tensor(temporal_horizons).float()
            static_data_t = self._check_tensor(static_data).float()

            yt = self(static_data_t, temporal_data_t, temporal_horizons_t)

            if self.task_type == "classification":
                return np.argmax(yt.cpu().numpy(), -1)
            else:
                return yt.cpu().numpy()

    def score(
        self,
        static_data: np.ndarray,
        temporal_data: np.ndarray,
        temporal_horizons: np.ndarray,
        outcome: np.ndarray,
    ) -> float:
        y_pred = self.predict(static_data, temporal_data, temporal_horizons)
        if self.task_type == "classification":
            return np.mean(y_pred == outcome)
        else:
            return np.mean(np.inner(outcome - y_pred, outcome - y_pred) / 2.0)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        static_data: np.ndarray,
        temporal_data: np.ndarray,
        temporal_horizons: np.ndarray,
        outcome: np.ndarray,
    ) -> Any:
        temporal_data_t = self._check_tensor(temporal_data).float()
        temporal_horizons_t = self._check_tensor(temporal_horizons).float()
        static_data_t = self._check_tensor(static_data).float()
        outcome_t = self._check_tensor(outcome).float()
        if self.task_type == "classification":
            outcome_t = outcome_t.long()

        return self._train(
            static_data_t, temporal_data_t, temporal_horizons_t, outcome_t
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _train(
        self,
        static_data: Optional[torch.Tensor],
        temporal_data: torch.Tensor,
        temporal_horizons: torch.Tensor,
        outcome: torch.Tensor,
    ) -> Any:
        loader = self.dataloader(static_data, temporal_data, temporal_horizons, outcome)
        # training and testing
        for it in range(self.n_iter):
            loss = self._train_epoch(loader)
            if it % self.n_iter_print == 0:
                log.info(f"Epoch:{it}| train loss: {loss}")

        return self

    def _train_epoch(self, loader: DataLoader) -> float:
        losses = []
        for step, (static_mb, temporal_mb, horizons_mb, y_mb) in enumerate(loader):
            self.optimizer.zero_grad()  # clear gradients for this training step

            pred = self(static_mb, temporal_mb, horizons_mb)  # rnn output
            loss = self.loss(pred, y_mb)

            loss.backward()  # backpropagation, compute gradients
            if self.clipping_value > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)
            self.optimizer.step()  # apply gradients

            losses.append(loss.detach().cpu())

        return np.mean(losses)

    def dataloader(
        self,
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        temporal_horizons: torch.Tensor,
        outcome: torch.Tensor,
    ) -> DataLoader:
        dataset = TensorDataset(static_data, temporal_data, temporal_horizons, outcome)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=self.dataloader_sampler,
            pin_memory=False,
        )

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)
