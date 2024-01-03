# stdlib
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

# third party
import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss
from pydantic import validate_arguments
from scipy.stats import multivariate_normal
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import MinMaxScaler, label_binarize
from torch import nn
from tqdm import tqdm

# synthcity absolute
from synthcity.logger import logger as log
from synthcity.plugins.core.models import bnaf

# Synthcity absolute
from synthcity.utils.constants import DEVICE


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_frequency(
    X_gt: pd.DataFrame, X_synth: pd.DataFrame, n_histogram_bins: int = 10
) -> dict:
    """Get percentual frequencies for each possible real categorical value.

    Returns:
        The observed and expected frequencies (as a percent).
    """
    res = {}
    for col in X_gt.columns:
        local_bins = min(n_histogram_bins, len(X_gt[col].unique()))

        if len(X_gt[col].unique()) < 5:  # categorical
            gt = (X_gt[col].value_counts() / len(X_gt)).to_dict()
            synth = (X_synth[col].value_counts() / len(X_synth)).to_dict()
        else:
            gt_vals, bins = np.histogram(X_gt[col], bins=local_bins)
            synth_vals, _ = np.histogram(X_synth[col], bins=bins)
            gt = {k: v / (sum(gt_vals) + 1e-8) for k, v in zip(bins, gt_vals)}
            synth = {k: v / (sum(synth_vals) + 1e-8) for k, v in zip(bins, synth_vals)}

        for val in gt:
            if val not in synth or synth[val] == 0:
                synth[val] = 1e-11
        for val in synth:
            if val not in gt or gt[val] == 0:
                gt[val] = 1e-11

        if gt.keys() != synth.keys():
            raise ValueError(f"Invalid features. {gt.keys()}. syn = {synth.keys()}")
        res[col] = (list(gt.values()), list(synth.values()))

    return res


def get_features(X: pd.DataFrame, sensitive_features: List[str] = []) -> List:
    """Return the non-sensitive features from dataset X"""
    features = list(X.columns)
    for col in sensitive_features:
        if col in features:
            features.remove(col)

    return features


def get_y_pred_proba_hlpr(y_pred_proba: np.ndarray, nclasses: int) -> np.ndarray:
    if nclasses == 2:
        if len(y_pred_proba.shape) < 2:
            return y_pred_proba

        if y_pred_proba.shape[1] == 2:
            return y_pred_proba[:, 1]

    return y_pred_proba


def evaluate_auc(
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
    classes: Union[np.ndarray, None] = None,
) -> Tuple[float, float]:
    y_test = np.asarray(y_test)
    y_pred_proba = np.asarray(y_pred_proba)

    nnan = sum(np.ravel(np.isnan(y_pred_proba)))

    if nnan:
        raise ValueError("nan in predictions. aborting")

    n_classes = len(set(np.ravel(y_test)))

    y_pred_proba_tmp = get_y_pred_proba_hlpr(y_pred_proba, n_classes)

    if n_classes > 2:
        fpr = dict()
        tpr = dict()
        precision = dict()
        recall = dict()
        average_precision = dict()
        roc_auc: dict = dict()

        if classes is None:
            classes = sorted(set(np.ravel(y_test)))

        y_test = label_binarize(y_test, classes=classes)

        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_test.ravel(), y_pred_proba_tmp.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_test.ravel(), y_pred_proba_tmp.ravel()
        )

        average_precision["micro"] = average_precision_score(
            y_test, y_pred_proba_tmp, average="micro"
        )

        aucroc = roc_auc["micro"]
        aucprc = average_precision["micro"]
    else:
        aucroc = roc_auc_score(np.ravel(y_test), y_pred_proba_tmp, multi_class="ovr")
        aucprc = average_precision_score(np.ravel(y_test), y_pred_proba_tmp)

    return aucroc, aucprc


# Probability Density Function Classes for Domias


class gaussian:  # TODO: rename with capital letters
    def __init__(self, X: np.ndarray) -> None:
        var = np.std(X, axis=0) ** 2
        mean = np.mean(X, axis=0)
        self.rv = multivariate_normal(mean, np.diag(var))

    def pdf(self, Z: np.ndarray) -> np.ndarray:
        return self.rv.pdf(Z)


class normal_func:  # TODO: rename with capital letters
    def __init__(self, X: np.ndarray) -> None:
        self.var = np.ones_like(np.std(X, axis=0) ** 2)
        self.mean = np.zeros_like(np.mean(X, axis=0))

    def pdf(self, Z: np.ndarray) -> np.ndarray:
        return multivariate_normal.pdf(Z, self.mean, np.diag(self.var))
        # return multivariate_normal.pdf(Z, np.zeros_like(self.mean), np.diag(np.ones_like(self.var)))


class normal_func_feat:  # TODO: rename with capital letters
    def __init__(
        self,
        X: np.ndarray,
        continuous: list,
    ) -> None:
        if np.any(np.array(continuous) > 1) or len(continuous) != X.shape[1]:
            raise ValueError("Continous variable needs to be boolean")
        self.feat = np.array(continuous).astype(bool)

        if np.sum(self.feat) == 0:
            raise ValueError("there needs to be at least one continuous feature")

        for i in np.arange(X.shape[1])[self.feat]:
            if len(np.unique(X[:, i])) < 10:
                log.debug(f"Warning: feature {i} does not seem continous. CHECK")

        self.var = np.std(X[:, self.feat], axis=0) ** 2
        self.mean = np.mean(X[:, self.feat], axis=0)

    def pdf(self, Z: np.ndarray) -> np.ndarray:
        return multivariate_normal.pdf(Z[:, self.feat], self.mean, np.diag(self.var))


class GeneratorInterface(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "GeneratorInterface":
        ...

    @abstractmethod
    def generate(self, count: int) -> pd.DataFrame:
        ...


# Domias helper functions
def compute_wd(
    X_syn: np.ndarray,
    X: np.ndarray,
) -> float:
    X_ = X.copy()
    X_syn_ = X_syn.copy()
    if len(X_) > len(X_syn_):
        X_syn_ = np.concatenate(
            [X_syn_, np.zeros((len(X_) - len(X_syn_), X_.shape[1]))]
        )

    scaler = MinMaxScaler().fit(X_)

    X_ = scaler.transform(X_)
    X_syn_ = scaler.transform(X_syn_)

    X_ten = torch.from_numpy(X_).reshape(-1, 1)
    Xsyn_ten = torch.from_numpy(X_syn_).reshape(-1, 1)
    OT_solver = SamplesLoss(loss="sinkhorn")

    return OT_solver(X_ten, Xsyn_ten).cpu().numpy().item()


# BNAF for Domias
def load_dataset(
    data_train: Optional[np.ndarray] = None,
    data_valid: Optional[np.ndarray] = None,
    data_test: Optional[np.ndarray] = None,
    device: Any = DEVICE,
    batch_dim: int = 50,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    if data_train is not None:
        dataset_train = torch.utils.data.TensorDataset(
            torch.from_numpy(data_train).float().to(device)
        )
        if data_valid is None:
            log.debug("No validation set passed")
            data_valid = np.random.randn(*data_train.shape)
        if data_test is None:
            log.debug("No test set passed")
            data_test = np.random.randn(*data_train.shape)

        dataset_valid = torch.utils.data.TensorDataset(
            torch.from_numpy(data_valid).float().to(device)
        )

        dataset_test = torch.utils.data.TensorDataset(
            torch.from_numpy(data_test).float().to(device)
        )
    else:
        raise RuntimeError()

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_dim, shuffle=True
    )

    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_dim, shuffle=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_dim, shuffle=False
    )

    return data_loader_train, data_loader_valid, data_loader_test


def create_model(
    n_dims: int,
    n_flows: int = 5,
    n_layers: int = 3,
    hidden_dim: int = 32,
    residual: Optional[str] = "gated",  # [None, "normal", "gated"]
    verbose: bool = False,
    device: Any = DEVICE,
    batch_dim: int = 50,
) -> nn.Module:
    flows: List = []
    for f in range(n_flows):
        layers: List = []
        for _ in range(n_layers - 1):
            layers.append(
                bnaf.MaskedWeight(
                    n_dims * hidden_dim,
                    n_dims * hidden_dim,
                    dim=n_dims,
                )
            )
            layers.append(bnaf.Tanh())

        flows.append(
            bnaf.BNAF(
                *(
                    [
                        bnaf.MaskedWeight(n_dims, n_dims * hidden_dim, dim=n_dims),
                        bnaf.Tanh(),
                    ]
                    + layers
                    + [bnaf.MaskedWeight(n_dims * hidden_dim, n_dims, dim=n_dims)]
                ),
                res=residual if f < n_flows - 1 else None,
            )
        )

        if f < n_flows - 1:
            flows.append(bnaf.Permutation(n_dims, "flip"))

    model = bnaf.Sequential(*flows).to(device)

    return model


def save_model(
    model: nn.Module,
    optimizer: Any,
    epoch: int,
    save: bool = False,
    workspace: Path = Path("workspace"),
) -> Callable:
    workspace.mkdir(parents=True, exist_ok=True)

    def f() -> None:
        if save:
            log.debug("Saving model..")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                workspace / "DomiasMIA_bnaf_checkpoint.pt",
            )

    return f


def load_model(
    model: nn.Module,
    optimizer: Any,
    workspace: Path = Path("workspace"),
) -> Callable:
    def f() -> None:
        if workspace.exists():
            return

        log.info("Loading model..")
        if (workspace / "checkpoint.pt").exists():
            checkpoint = torch.load(workspace / "checkpoint.pt")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

    return f


def compute_log_p_x(model: nn.Module, x_mb: torch.Tensor) -> torch.Tensor:
    y_mb, log_diag_j_mb = model(x_mb)
    log_p_y_mb = (
        torch.distributions.Normal(torch.zeros_like(y_mb), torch.ones_like(y_mb))
        .log_prob(y_mb)
        .sum(-1)
    )
    return log_p_y_mb + log_diag_j_mb


def train(
    model: nn.Module,
    optimizer: Any,
    scheduler: Any,
    data_loader_train: torch.utils.data.DataLoader,
    data_loader_valid: torch.utils.data.DataLoader,
    data_loader_test: torch.utils.data.DataLoader,
    workspace: Path = Path("workspace"),
    start_epoch: int = 0,
    device: Any = DEVICE,
    epochs: int = 50,
    save: bool = False,
    clip_norm: float = 0.1,
) -> Callable:
    epoch = start_epoch
    for epoch in range(start_epoch, start_epoch + epochs):
        t = tqdm(data_loader_train, smoothing=0, ncols=80, disable=True)
        train_loss: torch.Tensor = []

        for (x_mb,) in t:
            loss = -compute_log_p_x(model, x_mb).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            optimizer.step()
            optimizer.zero_grad()

            t.set_postfix(loss="{:.2f}".format(loss.item()), refresh=False)
            train_loss.append(loss)

        train_loss = torch.stack(train_loss).mean()
        optimizer.swap()
        validation_loss = -torch.stack(
            [
                compute_log_p_x(model, x_mb).mean().detach()
                for x_mb, in data_loader_valid
            ],
            -1,
        ).mean()
        optimizer.swap()

        log.debug(
            "Epoch {:3}/{:3} -- train_loss: {:4.3f} -- validation_loss: {:4.3f}".format(
                epoch + 1,
                start_epoch + epochs,
                train_loss.item(),
                validation_loss.item(),
            )
        )

        stop = scheduler.step(
            validation_loss,
            callback_best=save_model(
                model, optimizer, epoch + 1, save=save, workspace=workspace
            ),
            callback_reduce=load_model(model, optimizer, workspace=workspace),
        )

        if stop:
            break

    load_model(model, optimizer, workspace=workspace)()
    optimizer.swap()
    validation_loss = -torch.stack(
        [compute_log_p_x(model, x_mb).mean().detach() for x_mb, in data_loader_valid],
        -1,
    ).mean()
    test_loss = -torch.stack(
        [compute_log_p_x(model, x_mb).mean().detach() for x_mb, in data_loader_test], -1
    ).mean()

    log.debug(
        f"""
###### Stop training after {epoch + 1} epochs!
Validation loss: {validation_loss.item():4.3f}
Test loss:       {test_loss.item():4.3f}
"""
    )

    if save:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            workspace / "checkpoint.pt",
        )
        log.debug(
            f"""
###### Stop training after {epoch + 1} epochs!
Validation loss: {validation_loss.item():4.3f}
Test loss:       {test_loss.item():4.3f}
"""
        )

    def p_func(x: np.ndarray) -> np.ndarray:
        return np.exp(compute_log_p_x(model, x))

    return p_func


def density_estimator_trainer(
    data_train: np.ndarray,
    data_val: Optional[np.ndarray] = None,
    data_test: Optional[np.ndarray] = None,
    batch_dim: int = 50,
    flows: int = 5,
    layers: int = 3,
    hidden_dim: int = 32,
    residual: Optional[str] = "gated",  # [None, "normal", "gated"]
    workspace: Path = Path("workspace"),
    decay: float = 0.5,
    patience: int = 20,
    cooldown: int = 10,
    min_lr: float = 5e-4,
    early_stopping: int = 100,
    device: Any = DEVICE,
    epochs: int = 50,
    learning_rate: float = 1e-2,
    clip_norm: float = 0.1,
    polyak: float = 0.998,
    save: bool = True,
    load: bool = True,
) -> Tuple[Callable, nn.Module]:
    log.debug("Loading dataset..")
    data_loader_train, data_loader_valid, data_loader_test = load_dataset(
        data_train,
        data_val,
        data_test,
        device=device,
        batch_dim=batch_dim,
    )

    if save:
        log.debug("Creating directory experiment..")
        workspace.mkdir(parents=True, exist_ok=True)

    log.debug("Creating BNAF model..")
    model = create_model(
        data_train.shape[1],
        batch_dim=batch_dim,
        n_flows=flows,
        n_layers=layers,
        hidden_dim=hidden_dim,
        verbose=True,
        device=device,
    )

    log.debug("Creating optimizer..")
    optimizer = bnaf.Adam(
        model.parameters(), lr=learning_rate, amsgrad=True, polyak=polyak
    )

    log.debug("Creating scheduler..")

    scheduler = bnaf.ReduceLROnPlateau(
        optimizer,
        factor=decay,
        patience=patience,
        cooldown=cooldown,
        min_lr=min_lr,
        verbose=True,
        early_stopping=early_stopping,
        threshold_mode="abs",
    )

    if load:
        load_model(model, optimizer, workspace=workspace)()

    log.debug("Training..")
    p_func = train(
        model,
        optimizer,
        scheduler,
        data_loader_train,
        data_loader_valid,
        data_loader_test,
        workspace=workspace,
        device=device,
        epochs=epochs,
        save=save,
        clip_norm=clip_norm,
    )
    return p_func, model


def compute_metrics_baseline(
    y_scores: np.ndarray, y_true: np.ndarray, sample_weight: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    y_pred = y_scores > np.median(y_scores)
    y_true = np.nan_to_num(y_true)
    y_pred = np.nan_to_num(y_pred)
    y_scores = np.nan_to_num(y_scores)
    acc = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    auc = roc_auc_score(y_true, y_scores, sample_weight=sample_weight)

    return acc, auc
