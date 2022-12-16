# stdlib
from typing import Any, List, Optional, Union

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
from synthcity.plugins.core.models.tabular_vae import TabularVAE
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class TVAEPlugin(Plugin):
    """Tabular VAE implementation.

    Args:
        decoder_n_layers_hidden: int
            Number of hidden layers in the decoder
        decoder_n_units_hidden: int
            Number of hidden units in each layer of the decoder
        decoder_nonlin: string, default 'leaky_relu'
            Nonlinearity to use in the decoder. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        decoder_dropout: float
            Dropout value. If 0, the dropout is not used.
        encoder_n_layers_hidden: int
            Number of hidden layers in the encoder
        encoder_n_units_hidden: int
            Number of hidden units in each layer of the encoder
        encoder_nonlin: string, default 'leaky_relu'
            Nonlinearity to use in the encoder. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        encoder_dropout: float
            Dropout value for the encoder. If 0, the dropout is not used.
        n_iter: int
            Maximum number of iterations in the encoder.
        lr: float
            learning rate for optimizer. step_size equivalent in the JAX version.
        weight_decay: float
            l2 (ridge) penalty for the weights.
        batch_size: int
            Batch size
        random_state: int
            random_state used
        encoder_max_clusters: int
            The max number of clusters to create for continuous columns when encoding
        # early stopping
        n_iter_print: int
            Number of iterations after which to print updates and check the validation loss.
        n_iter_min: int
            Minimum number of iterations to go through before starting early stopping
        patience: int
            Max number of iterations without any improvement before early stopping is trigged.

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from synthcity.plugins import Plugins
        >>>
        >>> X, y = load_iris(as_frame = True, return_X_y = True)
        >>> X["target"] = y
        >>>
        >>> plugin = Plugins().get("tvae", n_iter = 100)
        >>> plugin.fit(X)
        >>>
        >>> plugin.generate(50)

    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_iter: int = 1000,
        n_units_conditional: int = 0,
        n_units_embedding: int = 500,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 500,
        random_state: int = 0,
        decoder_n_layers_hidden: int = 3,
        decoder_n_units_hidden: int = 500,
        decoder_nonlin: str = "leaky_relu",
        decoder_dropout: float = 0,
        encoder_n_layers_hidden: int = 3,
        encoder_n_units_hidden: int = 500,
        encoder_nonlin: str = "leaky_relu",
        encoder_dropout: float = 0.1,
        loss_factor: int = 1,
        data_encoder_max_clusters: int = 10,
        dataloader_sampler: Optional[sampler.Sampler] = None,
        clipping_value: int = 1,
        n_iter_print: int = 50,
        n_iter_min: int = 100,
        patience: int = 5,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.n_units_conditional = n_units_conditional
        self.n_units_embedding = n_units_embedding
        self.decoder_n_layers_hidden = decoder_n_layers_hidden
        self.decoder_n_units_hidden = decoder_n_units_hidden
        self.decoder_nonlin = decoder_nonlin
        self.decoder_dropout = decoder_dropout
        self.encoder_n_layers_hidden = encoder_n_layers_hidden
        self.encoder_n_units_hidden = encoder_n_units_hidden
        self.encoder_nonlin = encoder_nonlin
        self.encoder_dropout = encoder_dropout
        self.n_iter = n_iter
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.random_state = random_state
        self.data_encoder_max_clusters = data_encoder_max_clusters
        self.dataloader_sampler = dataloader_sampler
        self.loss_factor = loss_factor
        self.clipping_value = clipping_value

        self.n_iter_print = n_iter_print
        self.n_iter_min = n_iter_min
        self.patience = patience

    @staticmethod
    def name() -> str:
        return "tvae"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="n_iter", low=100, high=500, step=100),
            CategoricalDistribution(name="lr", choices=[1e-3, 2e-4, 1e-4]),
            IntegerDistribution(name="decoder_n_layers_hidden", low=1, high=5),
            CategoricalDistribution(name="weight_decay", choices=[1e-3, 1e-4]),
            CategoricalDistribution(name="batch_size", choices=[64, 128, 256, 512]),
            IntegerDistribution(name="n_units_embedding", low=50, high=500, step=50),
            IntegerDistribution(
                name="decoder_n_units_hidden", low=50, high=500, step=50
            ),
            CategoricalDistribution(
                name="decoder_nonlin", choices=["relu", "leaky_relu", "tanh", "elu"]
            ),
            FloatDistribution(name="decoder_dropout", low=0, high=0.2),
            IntegerDistribution(name="encoder_n_layers_hidden", low=1, high=5),
            IntegerDistribution(
                name="encoder_n_units_hidden", low=50, high=500, step=50
            ),
            CategoricalDistribution(
                name="encoder_nonlin",
                choices=["relu", "leaky_relu", "tanh", "elu"],
            ),
            FloatDistribution(name="encoder_dropout", low=0, high=0.2),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "TVAEPlugin":
        self.model = TabularVAE(
            X.dataframe(),
            n_units_embedding=self.n_units_embedding,
            n_units_conditional=self.n_units_conditional,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            n_iter=self.n_iter,
            decoder_n_layers_hidden=self.decoder_n_layers_hidden,
            decoder_n_units_hidden=self.decoder_n_units_hidden,
            decoder_nonlin=self.decoder_nonlin,
            decoder_nonlin_out_discrete="softmax",
            decoder_nonlin_out_continuous="tanh",
            decoder_residual=True,
            decoder_batch_norm=False,
            decoder_dropout=self.decoder_dropout,
            encoder_n_units_hidden=self.encoder_n_units_hidden,
            encoder_n_layers_hidden=self.encoder_n_layers_hidden,
            encoder_nonlin=self.encoder_nonlin,
            encoder_batch_norm=False,
            encoder_dropout=self.encoder_dropout,
            encoder_max_clusters=self.data_encoder_max_clusters,
            dataloader_sampler=self.dataloader_sampler,
            loss_factor=self.loss_factor,
            clipping_value=self.clipping_value,
            n_iter_min=self.n_iter_min,
            n_iter_print=self.n_iter_print,
            patience=self.patience,
        )
        self.model.fit(X.dataframe(), **kwargs)

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        cond: Optional[Union[pd.DataFrame, pd.Series]] = None
        if "cond" in kwargs:
            cond = np.asarray(kwargs["cond"])

        return self._safe_generate(self.model.generate, count, syn_schema, cond=cond)


plugin = TVAEPlugin
