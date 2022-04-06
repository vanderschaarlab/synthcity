# stdlib
from typing import Callable, List, Optional, Tuple

# third party
import numpy as np
import torch
from pydantic import validate_arguments
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# synthcity absolute
import synthcity.logger as log

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_nonlin(name: str) -> nn.Module:
    if name == "none":
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "softmax":
        return nn.Softmax(dim=-1)
    else:
        raise ValueError(f"Unknown nonlinearity {name}")


class LinearLayer(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_units_in: int,
        n_units_out: int,
        dropout: float = 0,
        batch_norm: bool = False,
        nonlin: Optional[str] = "relu",
    ) -> None:
        super(LinearLayer, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(n_units_in, n_units_out))

        if batch_norm:
            layers.append(nn.BatchNorm1d(n_units_out))

        if nonlin is not None:
            layers.append(get_nonlin(nonlin))

        self.model = nn.Sequential(*layers).to(DEVICE)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X.float()).to(DEVICE)


class ResidualLayer(LinearLayer):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_units_in: int,
        n_units_out: int,
        dropout: float = 0,
        batch_norm: bool = False,
        nonlin: Optional[str] = "relu",
    ) -> None:
        super(ResidualLayer, self).__init__(
            n_units_in,
            n_units_out,
            dropout=dropout,
            batch_norm=batch_norm,
            nonlin=nonlin,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self.model(X.float())
        return torch.cat([out, X], dim=1).to(DEVICE)


class MultiActivationHead(nn.Module):
    """Final layer with multiple activations. Useful for tabular data."""

    def __init__(self, activations: List[Tuple[nn.Module, int]]) -> None:
        super(MultiActivationHead, self).__init__()
        self.activations = []
        self.activation_lengths = []

        for activation, length in activations:
            self.activations.append(activation)
            self.activation_lengths.append(length)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[1] != np.sum(self.activation_lengths):
            raise RuntimeError(
                f"Shape mismatch for the activations: expected {np.sum(self.activation_lengths)}. Got shape {X.shape}."
            )

        split = 0
        for activation, step in zip(self.activations, self.activation_lengths):
            X[:, split : split + step] = activation(X[:, split : split + step])

            split += step

        return X


class MLP(nn.Module):
    """
    Basic neural net.

    Parameters
    ----------
    task_type: str
        classification or regression
    n_units_int: int
        Number of features
    n_units_out: int
        Number of outputs
    n_layers_hidden: int
        Number of hidden layers
    n_units_hidden: int
        Number of hidden units in each layer
    nonlin: string, default 'elu'
        Nonlinearity to use in NN. Can be 'elu', 'relu', 'selu', 'tanh' or 'leaky_relu'.
    lr: float
        learning rate for optimizer.
    weight_decay: float
        l2 (ridge) penalty for the weights.
    n_iter: int
        Maximum number of iterations.
    batch_size: int
        Batch size
    n_iter_print: int
        Number of iterations after which to print updates and check the validation loss.
    seed: int
        Seed used
    patience: int
        Number of iterations to wait before early stopping after decrease in validation loss
    n_iter_min: int
        Minimum number of iterations to go through before starting early stopping
    dropout: float
        Dropout value. If 0, the dropout is not used.
    clipping_value: int, default 1
        Gradients clipping value
    batch_norm: bool
        Enable/disable batch norm
    early_stopping: bool
        Enable/disable early stopping
    residual: bool
        Add residuals.
    loss: Callable
        Optional Custom loss function. If None, the loss is CrossEntropy for classification tasks, or RMSE for regression.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        task_type: str,  # classification/regression
        n_units_in: int,
        n_units_out: int,
        n_layers_hidden: int = 1,
        n_units_hidden: int = 100,
        nonlin: str = "relu",
        nonlin_out: Optional[List[Tuple[str, int]]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        opt_betas: tuple = (0.9, 0.999),
        n_iter: int = 1000,
        batch_size: int = 500,
        n_iter_print: int = 100,
        seed: int = 0,
        patience: int = 10,
        n_iter_min: int = 100,
        dropout: float = 0.1,
        clipping_value: int = 1,
        batch_norm: bool = True,
        early_stopping: bool = True,
        residual: bool = False,
        loss: Optional[Callable] = None,
    ) -> None:
        super(MLP, self).__init__()

        self.task_type = task_type
        self.seed = seed
        torch.manual_seed(seed)

        if residual:
            block = ResidualLayer
        else:
            block = LinearLayer

        # network
        layers = []

        if n_layers_hidden > 0:
            layers.append(
                block(n_units_in, n_units_hidden, batch_norm=batch_norm, nonlin=nonlin)
            )
            n_units_hidden += int(residual) * n_units_in

            # add required number of layers
            for i in range(n_layers_hidden - 1):
                layers.append(
                    block(
                        n_units_hidden,
                        n_units_hidden,
                        batch_norm=batch_norm,
                        nonlin=nonlin,
                        dropout=dropout,
                    )
                )
                n_units_hidden += int(residual) * n_units_hidden

            # add final layers
            layers.append(nn.Linear(n_units_hidden, n_units_out))
        else:
            layers = [nn.Linear(n_units_in, n_units_out)]

        if nonlin_out is not None:
            total_nonlin_len = 0
            activations = []
            for nonlin, nonlin_len in nonlin_out:
                total_nonlin_len += nonlin_len
                activations.append((get_nonlin(nonlin), nonlin_len))

            if total_nonlin_len != n_units_out:
                raise RuntimeError(
                    f"Shape mismatch for the output layer. Expected length {n_units_out}, but got {nonlin_out} with length {total_nonlin_len}"
                )
            layers.append(MultiActivationHead(activations))
        elif self.task_type == "classification":
            layers.append(MultiActivationHead([(nn.Softmax(dim=-1), n_units_out)]))

        self.model = nn.Sequential(*layers).to(DEVICE)

        # optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.opt_betas = opt_betas
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.opt_betas,
        )

        # training
        self.n_iter = n_iter
        self.n_iter_print = n_iter_print
        self.n_iter_min = n_iter_min
        self.batch_size = batch_size
        self.patience = patience
        self.clipping_value = clipping_value
        self.early_stopping = early_stopping
        if loss is not None:
            self.loss = loss
        else:
            if task_type == "classification":
                self.loss = nn.CrossEntropyLoss()
            else:
                self.loss = nn.MSELoss()

        torch.manual_seed(seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLP":
        Xt = self._check_tensor(X)
        yt = self._check_tensor(y)

        self._train(Xt, yt)

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task_type != "classification":
            raise ValueError(f"Invalid task type for predict_proba {self.task_type}")

        with torch.no_grad():
            Xt = self._check_tensor(X)

            yt = self.forward(Xt)

            return yt.cpu().numpy().squeeze()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            Xt = self._check_tensor(X)

            yt = self.forward(Xt)

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
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X.float())

    def _train_epoch(self, loader: DataLoader) -> float:
        train_loss = []

        for batch_ndx, sample in enumerate(loader):
            self.optimizer.zero_grad()

            X_next, y_next = sample
            if len(X_next) < 2:
                continue

            preds = self.forward(X_next).squeeze()

            batch_loss = self.loss(preds, y_next)

            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)

            self.optimizer.step()

            train_loss.append(batch_loss.detach())

        return torch.mean(torch.Tensor(train_loss))

    def _train(self, X: torch.Tensor, y: torch.Tensor) -> "MLP":
        X = self._check_tensor(X).float()
        y = self._check_tensor(y).squeeze().float()
        if self.task_type == "classification":
            y = y.long()

        # Load Dataset
        dataset = TensorDataset(X, y)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        loader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=False)

        # Setup the network and optimizer

        val_loss_best = 999999
        patience = 0

        # do training
        for i in range(self.n_iter):
            train_loss = self._train_epoch(loader)

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():
                    X_val, y_val = test_dataset.dataset.tensors

                    preds = self.forward(X_val).squeeze()
                    val_loss = self.loss(preds, y_val)

                    if self.early_stopping:
                        if val_loss_best > val_loss:
                            val_loss_best = val_loss
                            patience = 0
                        else:
                            patience += 1

                        if patience > self.patience and i > self.n_iter_min:
                            break

                    if i % self.n_iter_print == 0:
                        log.debug(
                            f"Epoch: {i}, loss: {val_loss}, train_loss: {train_loss}"
                        )

        return self

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(DEVICE)
        else:
            return torch.from_numpy(np.asarray(X)).to(DEVICE)

    def __len__(self) -> int:
        return len(self.model)
