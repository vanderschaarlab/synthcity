"""
Reference: Kotelnikov, Akim et al. “TabDDPM: Modelling Tabular Data with Diffusion Models.” ArXiv abs/2209.15421 (2022): n. pag.
"""

# stdlib
from pathlib import Path
from typing import Any, List, Sequence

# third party
import numpy as np
import pandas as pd

# Necessary packages
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import (
    Distribution,
    IntegerDistribution,
    LogDistribution,
    LogIntDistribution,
)
from synthcity.plugins.core.models.tabular_ddpm import TabDDPM
from synthcity.plugins.core.models.tabular_encoder import TabularEncoder
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.utils.callbacks import Callback
from synthcity.utils.constants import DEVICE


class TabDDPMPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.generic.plugin_tab_ddpm.TabDDPMPlugin
        :parts: 1


    Tabular denoising diffusion probabilistic model.

    Args:
        is_classification: bool = False
            Whether the task is classification or regression.
        n_iter: int = 1000
            Number of epochs for training.
        lr: float = 0.002
            Learning rate.
        weight_decay: float = 1e-4
            L2 weight decay.
        batch_size: int = 1024
            Size of mini-batches.
        model_type: str = "mlp"
            Type of model to use. Either "mlp" or "resnet".
        num_timesteps: int = 1000
            Number of timesteps to use in the diffusion process.
        gaussian_loss_type: str = "mse"
            Type of loss to use for the Gaussian diffusion process. Either "mse" or "kl".
        scheduler: str = "cosine"
            The scheduler of forward process variance 'beta' to use. Either "cosine" or "linear".
        device: Any = DEVICE
            Device to use for training.
        callbacks: Sequence[Callback] = ()
            Callbacks to use during training.
        log_interval: int = 100
            Number of iterations between logging.
        print_interval: int = 500
            Number of iterations between printing.
        n_layers_hidden: int = 3
            Number of hidden layers in the MLP.
        dim_hidden: int = 256
            Number of hidden units per hidden layer in the MLP.
        dropout: float = 0.0
            Dropout rate.
        dim_embed: int = 128
            Dimensionality of the embedding space.
        random_state: int
            random seed to use
        workspace: Path.
            Optional Path for caching intermediary results.
        compress_dataset: bool. Default = False.
            Drop redundant features before training the generator.
        sampling_patience: int.
            Max inference iterations to wait for the generated data to match the training schema.

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from synthcity.plugins import Plugins
        >>> X, y = load_iris(as_frame=True, return_X_y=True)
        >>> X["target"] = y
        >>> plugin = Plugins().get("ddpm", n_iter=100, is_classification=True)
        >>> plugin.fit(X)
        >>> plugin.generate(50)

    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        *,
        is_classification: bool = False,
        n_iter: int = 1000,
        lr: float = 0.002,
        weight_decay: float = 1e-4,
        batch_size: int = 1024,
        model_type: str = "mlp",
        num_timesteps: int = 1000,
        gaussian_loss_type: str = "mse",
        scheduler: str = "cosine",
        device: Any = DEVICE,
        callbacks: Sequence[Callback] = (),
        log_interval: int = 100,
        print_interval: int = 500,
        # model params
        n_layers_hidden: int = 3,
        dim_hidden: int = 256,
        dropout: float = 0.0,
        dim_embed: int = 128,
        # core plugin arguments
        random_state: int = 0,
        workspace: Path = Path("workspace"),
        compress_dataset: bool = False,
        sampling_patience: int = 500,
        **kwargs: Any
    ) -> None:
        super().__init__(
            device=device,
            random_state=random_state,
            sampling_patience=sampling_patience,
            workspace=workspace,
            compress_dataset=compress_dataset,
            **kwargs
        )

        self.is_classification = is_classification

        mlp_params = dict(
            n_layers_hidden=n_layers_hidden, n_units_hidden=dim_hidden, dropout=dropout
        )

        self.model = TabDDPM(
            n_iter=n_iter,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            num_timesteps=num_timesteps,
            gaussian_loss_type=gaussian_loss_type,
            is_classification=is_classification,
            scheduler=scheduler,
            device=device,
            callbacks=callbacks,
            log_interval=log_interval,
            print_interval=print_interval,
            model_type=model_type,
            mlp_params=mlp_params,
            dim_embed=dim_embed,
        )

        self.encoder = TabularEncoder(
            continuous_encoder="quantile",
            categorical_encoder="passthrough",
            cont_encoder_params=dict(random_state=random_state),
            cat_encoder_params=dict(),
        )

    @staticmethod
    def name() -> str:
        return "ddpm"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        """
        Hyperparameter Search space (from the paper)
        ----------------------------------------------
        Learning rate           LogUniform[0.00001, 0.003]
        Batch size              Cat{256, 4096}
        Diffusion timesteps     Cat{100, 1000}
        Training iterations     Cat{5000, 10000, 20000}
        Number of MLP layers    Int{2, 4, 6, 8}
        MLP width of layers     Int{128, 256, 512, 1024}
        Proportion of samples   Float{0.25, 0.5, 1, 2, 4, 8}
        ----------------------------------------------
        Dropout                 0.0
        Scheduler               cosine (Nichol, 2021)
        Gaussian diffusion loss MSE
        """
        return [
            LogDistribution(name="lr", low=1e-5, high=1e-1),
            LogIntDistribution(name="batch_size", low=256, high=4096),
            IntegerDistribution(name="num_timesteps", low=10, high=1000),
            LogIntDistribution(name="n_iter", low=1000, high=10000),
            IntegerDistribution(name="n_layers_hidden", low=2, high=8),
            LogIntDistribution(name="dim_hidden", low=128, high=1024),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "TabDDPMPlugin":
        """Fit the model to the data.

        Optionally, a condition can be given as the keyword argument `cond`.

        If the task is classification, the target labels are automatically regarded as the condition, and no additional condition should be given.

        If the task is regression, the target variable is not specially treated. There is no condition by default, but can be given by the user, either as a column name or an array-like.
        """
        df = X.dataframe()
        self.feature_names = df.columns

        cond = kwargs.pop("cond", None)
        self.loss_history = None

        if args:
            raise ValueError("Only keyword arguments are allowed")

        if self.is_classification:
            if cond is not None:
                raise ValueError(
                    "cond is already given by the labels for classification"
                )
            df, cond = X.unpack()
            self._labels, self._cond_dist = np.unique(cond, return_counts=True)
            self._cond_dist = self._cond_dist / self._cond_dist.sum()
            self.target_name = cond.name

        df = self.encoder.fit_transform(df)

        if cond is not None:
            if type(cond) is str:
                cond = df[cond]
            self.expecting_conditional = True

        if cond is not None:
            cond = pd.Series(cond, index=df.index)

        # NOTE: cond may also be included in the dataframe
        self.model.fit(df, cond, **kwargs)
        self.loss_history = self.model.loss_history

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> DataLoader:
        cond = kwargs.pop("cond", None)

        if self.is_classification and cond is None:
            # randomly generate labels following the distribution of the training data
            cond = np.random.choice(self._labels, size=count, p=self._cond_dist)

        if cond is not None and len(cond) > count:
            raise ValueError("The length of cond is less than the required count")

        def callback(count):  # type: ignore
            df = self.model.generate(count, cond=cond)
            df = self.encoder.inverse_transform(df)
            if self.is_classification:
                df = df.join(pd.Series(cond, name=self.target_name))
            return df[self.feature_names]  # reorder columns

        return self._safe_generate(callback, count, syn_schema, **kwargs)


plugin = TabDDPMPlugin
