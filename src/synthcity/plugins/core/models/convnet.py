# stdlib
from typing import Any, Optional, Tuple

# third party
import numpy as np
import torch
from monai.networks.layers.factories import Act
from monai.networks.nets import Classifier, Discriminator, Generator
from pydantic import validate_arguments
from torch import nn

# synthcity absolute
import synthcity.logger as log
from synthcity.utils.constants import DEVICE
from synthcity.utils.reproducibility import enable_reproducible_results


def map_nonlin(nonlin: str) -> Act:
    if nonlin == "relu":
        return Act.RELU
    elif nonlin == "elu":
        return Act.ELU
    elif nonlin == "prelu":
        return Act.PRELU
    elif nonlin == "leaky_relu":
        return Act.LEAKYRELU
    elif nonlin == "sigmoid":
        return Act.SIGMOID
    elif nonlin == "softmax":
        return Act.SOFTMAX
    elif nonlin == "tanh":
        return Act.TANH

    raise ValueError(f"Unknown activation {nonlin}")


class ConvNet(nn.Module):
    """
    Wrapper for convolutional nets for classification and regression.

    Parameters
    ----------
    task_type: str
        classifier or regression
    model: nn.Module
        classification or regression model implementation
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
    random_state: int
        random_state used
    patience: int
        Number of iterations to wait before early stopping after decrease in validation loss
    n_iter_min: int
        Minimum number of iterations to go through before starting early stopping
    clipping_value: int, default 1
        Gradients clipping value
    early_stopping: bool
        Enable/disable early stopping
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        task_type: str,  # classification/regression
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        opt_betas: tuple = (0.9, 0.999),
        n_iter: int = 1000,
        batch_size: int = 500,
        n_iter_print: int = 100,
        random_state: int = 0,
        patience: int = 10,
        n_iter_min: int = 100,
        clipping_value: int = 1,
        early_stopping: bool = True,
        device: Any = DEVICE,
    ) -> None:
        super(ConvNet, self).__init__()

        if task_type not in ["classification", "regression"]:
            raise ValueError(f"Invalid task type {task_type}")

        enable_reproducible_results(random_state)

        self.task_type = task_type
        self.device = device
        self.model = model
        self.random_state = random_state

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
        if task_type == "classification":
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.MSELoss()

    def fit(self, X: torch.utils.data.Dataset) -> "ConvNet":
        train_size = int(0.8 * len(X))
        test_size = len(X) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            X, [train_size, test_size]
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, pin_memory=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=len(test_dataset)
        )

        # Setup the network and optimizer

        val_loss_best = 999999
        patience = 0

        # do training
        for i in range(self.n_iter):
            train_loss = self._train_epoch(train_loader)

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():
                    X_val, y_val = next(iter(test_loader))
                    X_val = self._check_tensor(X_val)
                    y_val = self._check_tensor(y_val).long()

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
                            f"Epoch: {i}, val loss: {val_loss}, train_loss: {train_loss}"
                        )

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        if self.task_type != "classification":
            raise ValueError(f"Invalid task type for predict_proba {self.task_type}")

        with torch.no_grad():
            Xt = self._check_tensor(X)

            yt = self.forward(Xt)

            return yt.cpu()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            Xt = self._check_tensor(X)

            yt = self.forward(Xt)

            if self.task_type == "classification":
                return torch.argmax(yt.cpu(), -1).squeeze()
            else:
                return yt.cpu()

    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        y_pred = self.predict(X)
        if self.task_type == "classification":
            return torch.mean(y_pred == y)
        else:
            return torch.mean(torch.inner(y - y_pred, y - y_pred) / 2.0)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self._check_tensor(X)

        return self.model(X.float())

    def _train_epoch(self, loader: torch.utils.data.DataLoader) -> float:
        train_loss = []

        for batch_ndx, sample in enumerate(loader):
            self.optimizer.zero_grad()

            X_next, y_next = sample

            X_next = self._check_tensor(X_next)
            y_next = self._check_tensor(y_next).long()

            if len(X_next) < 2:
                continue

            preds = self.forward(X_next).squeeze()

            batch_loss = self.loss(preds, y_next)

            batch_loss.backward()

            if self.clipping_value > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)

            self.optimizer.step()

            train_loss.append(batch_loss.detach())

        return torch.mean(torch.Tensor(train_loss))

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)

    def __len__(self) -> int:
        return len(self.model)


class ConditionalGenerator(nn.Module):
    """Wrapper for making existing CNN generator conditional. Useful for Conditional GANs

    Args:
        model: nn.Module
            Core model.
        n_channels: int
            Number of channels in images
        n_units_latent: int
            Noise size for the input
        cond: torch.Tensor
            The reference conditional
        cond_embedding_n_units_hidden: int
            Size of the conditional embedding layer
    """

    def __init__(
        self,
        model: nn.Module,
        n_channels: int,
        n_units_latent: int,
        cond: Optional[torch.Tensor] = None,
        cond_embedding_n_units_hidden: int = 100,
        device: Any = DEVICE,
    ) -> None:
        super(ConditionalGenerator, self).__init__()

        self.model = model
        self.cond = cond
        self.n_channels = n_channels
        self.n_units_latent = n_units_latent
        self.device = device

        self.label_conditioned_generator: Optional[nn.Module] = None
        if cond is not None:
            classes = torch.unique(self.cond)
            self.label_conditioned_generator = nn.Sequential(
                nn.Embedding(len(classes), cond_embedding_n_units_hidden),
                nn.Linear(cond_embedding_n_units_hidden, n_channels * n_units_latent),
            ).to(device)

    def forward(
        self, noise: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if cond is None and self.cond is not None:
            perm = torch.randperm(self.cond.size(0))
            cond = self.cond[perm[: len(noise)]]

        if self.label_conditioned_generator is not None and cond is not None:
            cond_emb = self.label_conditioned_generator(cond.long()).view(
                -1, self.n_units_latent, self.n_channels, 1
            )
            noise = torch.cat(
                (noise, cond_emb), dim=2
            )  # add another channel with the conditional
        return self.model(noise)


class ConditionalDiscriminator(nn.Module):
    """

    Args:
        model: nn.Module
            Core model.
        n_channels: int
            Number of channels in images
        height: int
            Image height
        width: int
            Image width
        cond: torch.Tensor
            The reference conditional
        cond_embedding_n_units_hidden: int
            Size of the conditional embedding layer
    """

    def __init__(
        self,
        model: nn.Module,
        n_channels: int,
        height: int,
        width: int,
        cond: Optional[torch.Tensor] = None,
        cond_embedding_n_units_hidden: int = 100,
        device: Any = DEVICE,
    ) -> None:
        super(ConditionalDiscriminator, self).__init__()
        self.model = model
        self.cond = cond

        self.n_channels = n_channels
        self.height = height
        self.width = width

        self.device = device

        self.label_conditioned_generator: Optional[nn.Module] = None
        if cond is not None:
            classes = torch.unique(self.cond)
            self.label_conditioned_generator = nn.Sequential(
                nn.Embedding(len(classes), cond_embedding_n_units_hidden),
                nn.Linear(cond_embedding_n_units_hidden, n_channels * height * width),
            ).to(device)

    def forward(
        self, X: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if cond is None and self.cond is not None:
            perm = torch.randperm(self.cond.size(0))
            cond = self.cond[perm[: len(X)]]

        if self.label_conditioned_generator is not None and cond is not None:
            cond_emb = self.label_conditioned_generator(cond.long()).view(
                -1, self.n_channels, self.height, self.width
            )
            X = torch.cat(
                (X, cond_emb), dim=1
            )  # add another channel with the conditional

        return self.model(X)


# TODO: add more image processing architectures for better performance
def suggest_image_generator_discriminator_arch(
    n_units_latent: int,
    n_channels: int,
    height: int,
    width: int,
    generator_dropout: float = 0.2,
    generator_nonlin: str = "prelu",
    generator_n_residual_units: int = 2,
    discriminator_dropout: float = 0.2,
    discriminator_nonlin: str = "prelu",
    discriminator_n_residual_units: int = 2,
    device: Any = DEVICE,
    strategy: str = "predefined",
    cond: Optional[torch.Tensor] = None,
    cond_embedding_n_units_hidden: int = 100,
) -> Tuple[ConditionalGenerator, ConditionalDiscriminator]:
    """Helper for selecting compatible architecture for image generators and discriminators.

    Args:
        n_units_latent: int,
            Input size for the generator
        n_channels: int
            Number of channels in the image
        height: int
            Image height
        width: int
            Image width
        generator_dropout: float = 0.2
            Dropout value for the generator
        generator_nonlin: str
            name of the activation activation layers in the generator. Can be relu, elu, prelu or leaky_relu
        generator_n_residual_units: int
             integer stating number of convolutions in residual units for the generator, 0 means no residual units
        discriminator_dropout: float = 0.2
            Dropout value for the discriminator
        discriminator_nonlin: str
            name of the activation activation layers in the discriminator. Can be relu, elu, prelu or leaky_relu
        discriminator_n_residual_units: int
             integer stating number of convolutions in residual units for the discriminator, 0 means no residual units
        device: str
            PyTorch device. cpu, cuda
        strategy: str
            Which suggestion to use. Options:
                - predefined: a few hardcoded architectures for certain image shapes.
                - ...
    """
    cond_weight = 1 if cond is None else 2
    if strategy == "predefined":
        if height == 32 and width == 32:
            start_shape_gen = 4
            start_stride_disc = 2
        elif height == 64 and width == 64:
            start_shape_gen = 8
            start_stride_disc = 4
        elif height == 128 and width == 128:
            start_shape_gen = 16
            start_stride_disc = 8
        else:
            raise ValueError(
                f"Unsupported predefined arch : ({n_channels}, {height}, {width})"
            )

        generator = nn.Sequential(
            Generator(
                latent_shape=(n_units_latent, cond_weight * n_channels),
                start_shape=(64, start_shape_gen, start_shape_gen),
                channels=[64, 32, 16, n_channels],
                strides=[2, 2, 2, 1],
                kernel_size=3,
                dropout=generator_dropout,
                act=map_nonlin(generator_nonlin),
                num_res_units=generator_n_residual_units,
            ),
            nn.Tanh(),
        ).to(device)
        discriminator = Discriminator(
            in_shape=(cond_weight * n_channels, height, width),
            channels=[16, 32, 64, 1],
            strides=[start_stride_disc, 2, 2, 2],
            kernel_size=3,
            last_act=None,
            dropout=discriminator_dropout,
            act=map_nonlin(generator_nonlin),
            num_res_units=discriminator_n_residual_units,
        ).to(device)

        return ConditionalGenerator(
            model=generator,
            n_channels=n_channels,
            n_units_latent=n_units_latent,
            cond=cond,
            cond_embedding_n_units_hidden=cond_embedding_n_units_hidden,
            device=device,
        ), ConditionalDiscriminator(
            discriminator,
            n_channels=n_channels,
            height=height,
            width=width,
            cond=cond,
            cond_embedding_n_units_hidden=cond_embedding_n_units_hidden,
            device=device,
        )

    raise ValueError(f"unsupported image arch : ({n_channels}, {height}, {width})")


def suggest_image_classifier_arch(
    n_channels: int,
    height: int,
    width: int,
    classes: int,
    n_residual_units: int = 2,
    nonlin: str = "prelu",
    dropout: float = 0.2,
    last_nonlin: str = "softmax",
    device: Any = DEVICE,
    strategy: str = "predefined",
    # training
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    opt_betas: tuple = (0.9, 0.999),
    n_iter: int = 1000,
    batch_size: int = 500,
    n_iter_print: int = 100,
    random_state: int = 0,
    patience: int = 10,
    n_iter_min: int = 100,
    clipping_value: int = 1,
    early_stopping: bool = True,
) -> ConvNet:
    """Helper for selecting compatible architecture for image classifiers.

    Args:
        n_channels: int
            Number of channels in the image
        height: int
            Image height
        width: int
            Image width
        classes: int
            Number of output classes
        nonlin: str
            name of the activation activation layers. Can be relu, elu, prelu or leaky_relu
        last_act: str
            output activation
        dropout: float = 0.2
            Dropout value
        n_residual_units: int
             integer stating number of convolutions in residual units, 0 means no residual units
        device: str
            PyTorch device. cpu, cuda
     # Training
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
        random_state: int
            random_state used
        patience: int
            Number of iterations to wait before early stopping after decrease in validation loss
        n_iter_min: int
            Minimum number of iterations to go through before starting early stopping
        clipping_value: int, default 1
            Gradients clipping value
        early_stopping: bool
            Enable/disable early stopping


    """
    if strategy == "predefined":
        if height == 32 and width == 32:
            start_stride = 2
        elif height == 64 and width == 64:
            start_stride = 4
        elif height == 128 and width == 128:
            start_stride = 8
        else:
            raise ValueError(
                f"Unsupported predefined arch : ({n_channels}, {height}, {width})"
            )

        clf = Classifier(
            in_shape=(n_channels, height, width),
            classes=classes,
            channels=[16, 32, 64, 1],
            strides=[start_stride, 2, 2, 2],
            act=map_nonlin(nonlin),
            last_act=map_nonlin(last_nonlin),
            dropout=dropout,
            num_res_units=n_residual_units,
        ).to(device)
        return ConvNet(
            task_type="classification",
            model=clf,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            opt_betas=opt_betas,
            n_iter=n_iter,
            batch_size=batch_size,
            n_iter_print=n_iter_print,
            random_state=random_state,
            patience=patience,
            n_iter_min=n_iter_min,
            clipping_value=clipping_value,
            early_stopping=early_stopping,
        )

    raise ValueError(f"unsupported image arch : ({n_channels}, {height}, {width})")
