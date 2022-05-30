# stdlib
from typing import Any, Callable, Optional

# third party
import numpy as np
import torch
from pydantic import validate_arguments
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, sampler

# synthcity absolute
import synthcity.logger as log
from synthcity.utils.constants import DEVICE
from synthcity.utils.reproducibility import enable_reproducible_results


class WindowLinearLayer(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_units_in: int,
        window_size: int,
        n_units_out: int,
        device: Any = DEVICE,
    ) -> None:
        super(WindowLinearLayer, self).__init__()

        self.device = device
        self.window_size = window_size
        self.model = nn.Linear(n_units_in * window_size, n_units_out)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_feats = X.shape
        batch = X[:, seq_len - self.window_size :, :].reshape(
            batch_size, n_feats * self.window_size
        )

        return self.model(batch).to(self.device)


class RNN(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        task_type: str,  # regression, classification
        n_units_in: int,
        n_units_out: int,
        n_units_hidden: int = 100,
        n_layers: int = 2,
        n_iter: int = 100,
        mode: str = "RNN",  # RNN, LSTM, GRU
        n_iter_print: int = 10,
        batch_size: int = 150,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        window_size: int = 1,
        device: Any = DEVICE,
        dataloader_sampler: Optional[sampler.Sampler] = None,
        loss: Optional[Callable] = None,
        seed: int = 0,
    ) -> None:
        super(RNN, self).__init__()

        enable_reproducible_results(seed)
        assert task_type in ["classification", "regression"]
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
        self.n_units_in = n_units_in
        self.n_units_out = n_units_out
        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.device = device
        self.window_size = window_size
        self.dataloader_sampler = dataloader_sampler

        if mode == "RNN":
            self.rnn = nn.RNN(
                input_size=self.n_units_in,
                hidden_size=self.n_units_hidden,
                num_layers=self.n_layers,
                batch_first=True,
            )
        elif mode == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.n_units_in,
                hidden_size=self.n_units_hidden,
                num_layers=self.n_layers,
                batch_first=True,
            )
        elif mode == "GRU":
            self.rnn = nn.GRU(
                input_size=self.n_units_in,
                hidden_size=self.n_units_hidden,
                num_layers=self.n_layers,
                batch_first=True,
            )
        else:
            raise ValueError(f"unsupported mode {mode}")
        self.mode = mode

        self.out = nn.Sequential(
            WindowLinearLayer(self.n_units_hidden, self.window_size, self.n_units_out),
        )
        if task_type == "classification":
            self.out.append(nn.Softmax(dim=-1))

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )  # optimize all rnn parameters

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)

        if self.mode == "LSTM":
            r_out, (h_n, h_c) = self.rnn(
                x, None
            )  # None represents zero initial hidden state
        else:
            r_out, h_n = self.rnn(x, None)

        # choose r_out at the last <window size> steps
        return self.out(r_out).reshape(self.output_shape)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X_test = self._check_tensor(X).float()

            yt = self(X_test)

            if self.task_type == "classification":
                return np.argmax(yt.cpu().numpy().squeeze(), -1).squeeze()
            else:
                return yt.cpu().numpy().squeeze()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        if self.task_type == "classification":
            return np.mean(y_pred == y)
        else:
            return np.mean(np.inner(y - y_pred, y - y_pred) / 2.0)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        Xt = self._check_tensor(X).float()
        yt = self._check_tensor(y).float()
        if self.task_type == "classification":
            yt = yt.long()

        return self._train(Xt, yt)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _train(self, X: torch.Tensor, y: torch.Tensor) -> Any:
        loader = self.dataloader(X, y)
        self.output_shape = list(y.shape)
        if self.task_type == "classification":
            self.output_shape.append(self.n_units_out)

        # training and testing
        for it in range(self.n_iter):

            loss = self._train_epoch(loader)
            if it % self.n_iter_print == 0:
                log.info(f"Epoch:{it}| train loss: {loss}")

        return self

    def _train_epoch(self, loader: DataLoader) -> float:
        losses = []
        for step, (Xmb, ymb) in enumerate(loader):
            self.optimizer.zero_grad()  # clear gradients for this training step

            pred = self(Xmb)  # rnn output
            loss = self.loss(pred, ymb)

            loss.backward()  # backpropagation, compute gradients
            self.optimizer.step()  # apply gradients

            losses.append(loss.detach().cpu())

        return np.mean(losses)

    def dataloader(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> DataLoader:
        dataset = TensorDataset(X, y)
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
