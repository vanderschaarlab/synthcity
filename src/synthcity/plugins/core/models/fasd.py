from typing import Any, Callable, List, Optional, Tuple

# third party packages
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# synthcity absolutes
from synthcity.utils.constants import DEVICE

# synthcity relatives
from .mlp import MLP


class FASD(nn.Module):

    def __init__(
        self,
        n_units_in: int,
        n_units_embedding: int,
        n_units_hidden: int,
        n_layers_hidden: int,
        hidden_nonlin: str,
        target_nonlin_out: List[Tuple[str, int]],
        device: Any = DEVICE,
        random_state: int = 0,
        batch_size: int = 200,
        n_iter: int = 300,
        patience: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.001,
        n_iter_min: int = 30,
        dropout: float = 0.1,
        clipping_value: int = 1,
    ) -> None:
        super(FASD, self).__init__()
        self.random_state = random_state
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.device = device
        torch.manual_seed(self.random_state)
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_iter_min = n_iter_min
        self.n_units_embedding = n_units_embedding

        # the nonlin out for the encoder is simply the hidden nonlin for each hidden node
        # this does force the embedding activation and encoder nonlinearity to be the same
        hidden_nonlin_out = []
        for _ in list(range(n_units_embedding)):
            hidden_nonlin_out.append((hidden_nonlin, 1))

        self.encoder = MLP(
            n_units_in=n_units_in,
            n_units_out=n_units_embedding,
            task_type="regression",
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            nonlin=hidden_nonlin,
            nonlin_out=hidden_nonlin_out,
            random_state=self.random_state,
            dropout=dropout,
            clipping_value=clipping_value,
            device=self.device,
        ).to(device=self.device)

        # output units can be deduced from target_nonlin_out
        for _, nonlin_len in target_nonlin_out:
            n_units_out = nonlin_len

        if n_units_out > 1:
            task_type = "classification"
            self.criterion = nn.CrossEntropyLoss()
        else:
            task_type = "regression"
            self.criterion = nn.MSELoss()
        self.predictor = MLP(
            n_units_in=n_units_embedding,
            n_units_out=n_units_out,
            task_type=task_type,
            n_layers_hidden=0,  # shallow predictor
            n_units_hidden=0,
            nonlin="none",
            nonlin_out=target_nonlin_out,
            random_state=random_state,
            dropout=0,
            clipping_value=clipping_value,
            device=self.device,
        ).to(device=self.device)

    def forward(self, X: Tensor):
        """Forward pass through the network"""
        Xt = self._check_tensor(X)
        Xt = self.encoder(Xt)
        Xt = self.predictor(Xt)
        return Xt

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Train the network to predict y from X"""

        self.target_cols = y.columns

        # stratified validation split
        X, X_val, y, y_val = train_test_split(
            X, y, stratify=y, train_size=0.8, random_state=self.random_state
        )

        # create tensor datasets and dataloaders
        Xt = self._check_tensor(X)
        Xt_val = self._check_tensor(X_val)
        yt = self._check_tensor(y)
        yt_val = self._check_tensor(y_val)

        loader = DataLoader(
            dataset=TensorDataset(Xt, yt), batch_size=self.batch_size, pin_memory=False
        )
        val_loader = DataLoader(
            dataset=TensorDataset(Xt_val, yt_val),
            batch_size=self.batch_size,
            pin_memory=False,
        )

        # perform training loop
        self._train(loader=loader, val_loader=val_loader)
        return self

    def _train(self, loader: DataLoader, val_loader: DataLoader):
        """Perform the training loop"""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_state_dict = None
        best_loss = float("inf")
        patience = 0
        for epoch in tqdm(range(self.n_iter)):
            self.train()
            train_loss = 0
            for inputs, targets in loader:
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            val_loss = self.validate(val_loader)

            if val_loss >= best_loss:
                patience += 1
            else:
                best_loss = val_loss
                best_state_dict = self.state_dict()
                patience = 0

            if patience >= self.patience and epoch >= self.n_iter_min:
                break

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

        return self

    def validate(self, val_loader):
        self.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def encode(self, X: pd.DataFrame):
        """Pass input features through the Encoder to get representations"""

        # turn into tensor
        Xt = self._check_tensor(X)

        # forward pass through encoder
        Xt = self.encoder(Xt)

        # turn into dataframe
        X_enc = pd.DataFrame(
            Xt.cpu().detach().numpy(),
            columns=["rep_" + str(x) for x in list(range(self.n_units_embedding))],
        )

        return X_enc

    def predict(self, X: pd.DataFrame):
        """Predict y from X (labels not probabilities)"""

        # turn into tensor
        Xt = self._check_tensor(X)

        # forward pass through predictor
        yt = self.predictor(Xt)

        # turn into dataframe
        y_enc = pd.DataFrame(yt.cpu().detach().numpy(), columns=self.target_cols)

        return y_enc

    def _check_tensor(self, X: Tensor) -> Tensor:
        if isinstance(X, Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)


class FASD_Decoder(nn.Module):

    def __init__(
        self,
        n_units_in: int,
        n_units_hidden: int,
        n_units_out: int,
        n_layers_hidden: int,
        nonlin: str,
        nonlin_out: List[Tuple[str, int]],
        device: Any = DEVICE,
        random_state: int = 0,
        batch_size: int = 200,
        n_iter: int = 300,
        patience: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.001,
        n_iter_min: int = 30,
        dropout: float = 0.1,
        clipping_value: int = 1,
    ) -> None:
        super(FASD_Decoder, self).__init__()
        self.random_state = random_state
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.device = device
        torch.manual_seed(self.random_state)
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_iter_min = n_iter_min
        self.nonlin_out = nonlin_out

        self.decoder = MLP(
            n_units_in=n_units_in,
            n_units_out=n_units_out,
            task_type="regression",
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            nonlin=nonlin,
            nonlin_out=nonlin_out,
            random_state=self.random_state,
            dropout=dropout,
            clipping_value=clipping_value,
            device=self.device,
        ).to(device=self.device)

    def forward(self, X: Tensor):
        """Forward pass through the Decoder"""
        Xt = self._check_tensor(X)
        Xt = self.decoder(Xt)
        return Xt

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Train the network to reconstruct input features from representations"""

        self.target_cols = y.columns

        # validation split (no stratification for original input data as target)
        X, X_val, y, y_val = train_test_split(
            X, y, train_size=0.8, random_state=self.random_state
        )

        # create tensor datasets and dataloaders
        Xt = self._check_tensor(X)
        Xt_val = self._check_tensor(X_val)
        yt = self._check_tensor(y)
        yt_val = self._check_tensor(y_val)

        loader = DataLoader(
            dataset=TensorDataset(Xt, yt), batch_size=self.batch_size, pin_memory=False
        )
        val_loader = DataLoader(
            dataset=TensorDataset(Xt_val, yt_val),
            batch_size=self.batch_size,
            pin_memory=False,
        )

        # perform training loop
        self._train(loader=loader, val_loader=val_loader)
        return self

    def _train(self, loader: DataLoader, val_loader: DataLoader):
        """Perform the training loop"""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_state_dict = None
        best_loss = float("inf")
        patience = 0
        for epoch in tqdm(range(self.n_iter)):
            self.train()
            train_loss = 0
            for inputs, targets in loader:
                outputs = self.forward(inputs)
                loss = self._loss_function_standard(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            val_loss = self.validate(val_loader)

            if val_loss >= best_loss:
                patience += 1
            else:
                best_loss = val_loss
                best_state_dict = self.state_dict()
                patience = 0

            if patience >= self.patience and epoch >= self.n_iter_min:
                break

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

        return self

    def validate(self, val_loader):
        self.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.forward(inputs)
                loss = self._loss_function_standard(outputs, targets)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def decode(self, X: pd.DataFrame):
        """Pass representations through the Decoder to get reconstructed input features"""

        # turn into tensor
        Xt = self._check_tensor(X)

        # forward pass through decoder
        Xt = self.decoder(Xt)

        # turn into dataframe
        X_enc = pd.DataFrame(
            Xt.cpu().detach().numpy(),
            columns=self.target_cols,
        )

        return X_enc

    def _loss_function_standard(
        self,
        reconstructed: Tensor,
        real: Tensor,
    ) -> Tensor:
        step = 0
        loss = []
        for activation, length in self.nonlin_out:
            step_end = step + length
            # reconstructed is after the activation
            if activation == "softmax":
                discr_loss = nn.NLLLoss(reduction="sum")(
                    torch.log(reconstructed[:, step:step_end] + 1e-8),
                    torch.argmax(real[:, step:step_end], dim=-1),
                )
                loss.append(discr_loss)
            else:
                diff = reconstructed[:, step:step_end] - real[:, step:step_end]
                cont_loss = (50 * diff**2).sum()

                loss.append(cont_loss)
            step = step_end

        reconstruction_loss = torch.sum(torch.stack(loss)) / real.shape[0]

        if torch.isnan(reconstruction_loss):
            raise RuntimeError("NaNs detected in the reconstruction_loss")

        return reconstruction_loss

    def _check_tensor(self, X: Tensor) -> Tensor:
        if isinstance(X, Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)
