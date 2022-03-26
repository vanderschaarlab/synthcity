"""Anonymization through Data Synthesis using Generative Adversarial Networks:
A harmonizing advancement for AI in medicine (ADS-GAN) Codebase.

Reference: Jinsung Yoon, Lydia N. Drumright, Mihaela van der Schaar,
"Anonymization through Data Synthesis using Generative Adversarial Networks (ADS-GAN):
A harmonizing advancement for AI in medicine,"
IEEE Journal of Biomedical and Health Informatics (JBHI), 2019.
Paper link: https://ieeexplore.ieee.org/document/9034117
"""
# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd

# Necessary packages
import torch
from sklearn.preprocessing import MinMaxScaler

# synthcity absolute
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.models import GAN


class AdsGAN:
    """Basic ADS-GAN framework."""

    def __init__(
        self,
        epochs: int = 100,  # the number of student training iterations
        discr_epochs: int = 1,  # the number of student training iterations
        batch_size: int = 64,  # the number of batch size for training student and generator
        learning_rate: float = 1e-4,
        alpha: int = 20,
        clipping_value: float = 0.01,
    ) -> None:
        self.epochs = epochs
        self.discr_epochs = discr_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.clipping_value = clipping_value
        self.encoder = MinMaxScaler()

    def fit(
        self,
        X_train: np.ndarray,
    ) -> "AdsGAN":
        X_train = self.encoder.fit_transform(X_train)

        X_train = torch.from_numpy(np.asarray(X_train))
        features = X_train.shape[1]

        self.model = GAN(
            n_features=features,
            n_units_latent=features,
            batch_size=self.batch_size,
            generator_n_layers_hidden=2,
            generator_n_units_hidden=4 * features,
            generator_nonlin="tanh",
            generator_nonlin_out="sigmoid",
            generator_lr=self.learning_rate,
            generator_residual=True,
            generator_n_iter=self.epochs,
            generator_batch_norm=False,
            generator_dropout=0,
            generator_weight_decay=1e-3,
            discriminator_n_units_hidden=4 * features,
            discriminator_n_iter=self.discr_epochs,
            discriminator_nonlin="leaky_relu",
            discriminator_batch_norm=False,
            discriminator_dropout=0.1,
            discriminator_lr=self.learning_rate,
            discriminator_weight_decay=1e-3,
            clipping_value=self.clipping_value,
        )
        self.model.fit(X_train)
        return self

    def sample(self, count: int) -> np.ndarray:
        with torch.no_grad():
            return self.encoder.inverse_transform(self.model.generate(count))


class AdsGANPlugin(Plugin):
    """AdsGAN plugin.

    Example:
        >>> from synthcity.plugins import Plugins
        >>> plugin = Plugins().get("adsgan")
        >>> from sklearn.datasets import load_iris
        >>> X = load_iris()
        >>> plugin.fit(X)
        >>> plugin.generate()
    """

    def __init__(
        self,
        epochs: int = 100,
        discr_epochs: int = 1,
        batch_size: int = 64,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.model = AdsGAN(
            epochs=epochs,
            discr_epochs=discr_epochs,
            batch_size=batch_size,
        )

    @staticmethod
    def name() -> str:
        return "adsgan"

    @staticmethod
    def type() -> str:
        return "gan"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "AdsGANPlugin":
        self.model.fit(X)

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.sample, count, syn_schema)


plugin = AdsGANPlugin
