# stdlib
from typing import Callable, Optional

# third party
import numpy as np
import torch
from pydantic import validate_arguments
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# synthcity absolute
import synthcity.logger as log

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 1e-8

NONLIN = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    """
    Basic neural net.

    Parameters
    ----------
    task_type: str
        classification or regression
    n_layers_hidden: int
        Number of hidden layers (n_layers_hidden x n_units_hidden + 1 x Linear layer)
    n_units_hidden: int
        Number of hidden units in each layer
    nonlin: string, default 'elu'
        Nonlinearity to use in NN. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
    lr: float
        learning rate for optimizer. step_size equivalent in the JAX version.
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
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        n_iter: int = 1000,
        batch_size: int = 64,
        n_iter_print: int = 100,
        seed: int = 0,
        patience: int = 10,
        n_iter_min: int = 100,
        dropout: float = 0.1,
        clipping_value: int = 1,
        batch_norm: bool = True,
        early_stopping: bool = True,
        loss: Optional[Callable] = None,
    ) -> None:
        super(MLP, self).__init__()

        if nonlin not in list(NONLIN.keys()):
            raise ValueError(f"Unknown nonlinearity {nonlin}")

        self.task_type = task_type
        self.seed = seed

        # network
        NL = NONLIN[nonlin]
        layers = []

        if n_layers_hidden > 0:
            layers.append(nn.Linear(n_units_in, n_units_hidden))
            if batch_norm:
                layers.append(nn.BatchNorm1d(n_units_hidden))
            layers.append(NL())

            # add required number of layers
            for i in range(n_layers_hidden - 1):
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

                layers.append(nn.Linear(n_units_hidden, n_units_hidden))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(n_units_hidden))
                layers.append(NL())

            # add final layers
            layers.append(nn.Linear(n_units_hidden, n_units_out))
        else:
            layers = [nn.Linear(n_units_in, n_units_out)]

        if self.task_type == "classification":
            layers.append(nn.Softmax(dim=-1))

        self.model = nn.Sequential(*layers).to(DEVICE)

        # optimizer
        self.lr = lr
        self.weight_decay = weight_decay

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
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        val_loss_best = 999999
        patience = 0

        # do training
        for i in range(self.n_iter):
            train_loss = []

            for batch_ndx, sample in enumerate(loader):
                optimizer.zero_grad()

                X_next, y_next = sample

                preds = self.forward(X_next).squeeze()

                batch_loss = self.loss(preds, y_next)

                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)

                optimizer.step()

                train_loss.append(batch_loss.detach())

            train_loss = torch.Tensor(train_loss).to(DEVICE)

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
                        log.info(
                            f"Epoch: {i}, loss: {val_loss}, train_loss: {torch.mean(train_loss)}"
                        )

        return self

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(DEVICE)
        else:
            return torch.from_numpy(np.asarray(X)).to(DEVICE)
