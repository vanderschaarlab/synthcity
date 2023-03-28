"""
Reference: "GOGGLE: Generative Modelling for Tabular Data by Learning Relational Structure" Authors: Tennison Liu, Zhaozhi Qian, Jeroen Berrevoets, Mihaela van der Schaar
"""

# stdlib
from pathlib import Path
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd

# Necessary packages
from pydantic import validate_arguments
from torch.utils.data import sampler

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.models.tabular_goggle import TabularGoggle
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE


class GOGGLEPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.generic.plugin_GOGGLE.GOGGLEPlugin
        :parts: 1

    Args:


    Example:
        >>> from sklearn.datasets import load_iris
        >>> from synthcity.plugins import Plugins
        >>>
        >>> X, y = load_iris(as_frame = True, return_X_y = True)
        >>> X["target"] = y
        >>>
        >>> plugin = Plugins().get("goggle", n_iter = 100)
        >>> plugin.fit(X)
        >>>
        >>> plugin.generate(50)

    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_iter: int = 1000,
        encoder_dim: int = 64,
        encoder_l: int = 2,
        het_encoding: bool = True,
        encoder_nonlin: str = "tanh",
        decoder_nonlin: str = "tanh",
        decoder_dim: int = 64,
        decoder_l: int = 2,
        data_encoder_max_clusters: int = 10,
        threshold: float = 0.1,
        decoder_arch: str = "gcn",
        graph_prior: Optional[np.ndarray] = None,
        prior_mask: Optional[np.ndarray] = None,
        alpha: float = 0.1,
        beta: float = 0.1,
        iter_opt: bool = True,
        learning_rate: float = 5e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 32,
        patience: int = 50,
        logging_epoch: int = 100,
        # core plugin arguments
        device: str = DEVICE,
        random_state: int = 0,
        sampling_patience: int = 500,
        workspace: Path = Path("workspace"),
        compress_dataset: bool = False,
        dataloader_sampler: Optional[sampler.Sampler] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            device=device,
            random_state=random_state,
            sampling_patience=sampling_patience,
            workspace=workspace,
            compress_dataset=compress_dataset,
            **kwargs,
        )

        # goggle model args
        self.encoder_dim = encoder_dim
        self.encoder_l = encoder_l
        self.het_encoding = het_encoding
        self.decoder_dim = decoder_dim
        self.decoder_l = decoder_l
        self.threshold = threshold
        self.data_encoder_max_clusters = data_encoder_max_clusters
        self.decoder_arch = decoder_arch
        self.graph_prior = graph_prior
        self.prior_mask = prior_mask
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        self.iter_opt = iter_opt
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.logging_epoch = logging_epoch
        self.dataloader_sampler = dataloader_sampler
        self.encoder_nonlin = encoder_nonlin
        self.decoder_nonlin = decoder_nonlin

    @staticmethod
    def name() -> str:
        return "goggle"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="n_iter", low=100, high=500, step=100),
            IntegerDistribution(name="encoder_dim", low=32, high=128, step=16),
            IntegerDistribution(name="encoder_l", low=1, high=5, step=1),
            IntegerDistribution(name="decoder_dim", low=32, high=128, step=16),
            FloatDistribution(name="threshold", low=0.0, high=1.0),
            CategoricalDistribution(
                name="decoder_arch", choices=["gcn", "het"]
            ),  # TODO: Is this needed here?
            CategoricalDistribution(
                name="iter_opt", choices=[True, False]
            ),  # TODO: Is this needed here?
            FloatDistribution(name="alpha", low=0, high=1.0),
            FloatDistribution(name="beta", low=0, high=1.0),
        ]

    # def _prepare_cond(
    #     self, cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray, list]]
    # ) -> Optional[np.ndarray]:
    #     if cond is None:
    #         return None

    #     cond = np.asarray(cond)
    #     if len(cond.shape) == 1:
    #         cond = cond.reshape(-1, 1)

    #     return cond

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "GOGGLEPlugin":
        # cond: Optional[Union[pd.DataFrame, pd.Series]] = None
        # if "cond" in kwargs:
        #     cond = self._prepare_cond(kwargs["cond"])
        self.model = TabularGoggle(
            X.dataframe(),
            # cond=cond,
            n_iter=self.n_iter,
            encoder_dim=self.encoder_dim,
            encoder_l=self.encoder_l,
            het_encoding=self.het_encoding,
            decoder_dim=self.decoder_dim,
            decoder_l=self.decoder_l,
            encoder_max_clusters=self.data_encoder_max_clusters,
            threshold=self.threshold,
            decoder_arch=self.decoder_arch,
            graph_prior=self.graph_prior,
            prior_mask=self.prior_mask,
            device=self.device,
            alpha=self.alpha,
            beta=self.beta,
            iter_opt=self.iter_opt,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            patience=self.patience,
            logging_epoch=self.logging_epoch,
            dataloader_sampler=self.dataloader_sampler,
            schema=self.schema(),
            encoder_nonlin=self.encoder_nonlin,
            decoder_nonlin=self.decoder_nonlin,
            **kwargs,
        )
        if "cond" in kwargs and kwargs["cond"] is not None:
            raise NotImplementedError(
                "conditional generation is not currently available for the goggle plugin."
            )
        self.model.fit(X.dataframe(), **kwargs)  # , cond=cond, **kwargs)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        # cond: Optional[Union[pd.DataFrame, pd.Series]] = None
        # if "cond" in kwargs and kwargs["cond"] is not None:
        #     cond = np.asarray(kwargs["cond"])
        if "cond" in kwargs and kwargs["cond"] is not None:
            raise NotImplementedError(
                "conditional generation is not currently available for the goggle plugin."
            )

        return self._safe_generate(
            self.model.generate, count, syn_schema
        )  # , cond=cond)


plugin = GOGGLEPlugin
