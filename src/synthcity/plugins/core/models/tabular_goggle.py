# stdlib
from typing import Any, Optional, Union

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments

# synthcity absolute
from synthcity.utils.constants import DEVICE

# synthcity relative
from .goggle import Goggle, GoggleLoss
from .tabular_encoder import TabularEncoder


class TabularGoggle:
    def __init__(
        self,
        X: pd.DataFrame,
        n_iter: int = 1000,
        encoder_dim: int = 64,
        encoder_l: int = 2,
        het_encoding: bool = True,
        decoder_dim: int = 64,
        decoder_l: int = 2,
        threshold: float = 0.1,
        decoder_arch: str = "gcn",
        graph_prior: Optional[np.ndarray] = None,
        prior_mask: Optional[np.ndarray] = None,
        device: Union[str, torch.device] = DEVICE,
        alpha: float = 0.1,
        beta: float = 0.1,
        iter_opt: bool = True,
        learning_rate: float = 5e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 32,
        patience: int = 50,
        dataloader_sampler: Any = None,
        logging_epoch: int = 100,
        encoder_nonlin: str = "relu",
        decoder_nonlin: str = "relu",
        encoder_max_clusters: int = 20,
        encoder_whitelist: list = [],
        decoder_nonlin_out_discrete: str = "softmax",
        decoder_nonlin_out_continuous: str = "tanh",
        random_state: int = 0,
    ):
        """
        .. inheritance-diagram:: synthcity.plugins.core.models.tabular_goggle.TabularGoggle
        :parts: 1


        Goggle for tabular data.

        This class combines Goggle and tabular encoder to form a generative model for tabular data.

        Args:
            X: pd.DataFrame
                Reference dataset, used for training the tabular encoder
            n_iter: int = 1000
                Maximum number of iterations in the Generator. Defaults to 1000.
            encoder_dim: int = 64
                The number of dimensions in goggle's encoder. Defaults to 64.
            encoder_l: int = 2
                Number of layers in goggle's encoder. Defaults to 2.
            het_encoding: bool = True
                Flag to use heterogeneous encoding in core goggle model. Defaults to True.
            decoder_dim: int = 64
                The number of dimensions in goggle's decoder. Defaults to 64.
            decoder_l: int = 2
                Number of layers in goggle's decoder. Defaults to 2.
            threshold: float = 0.1
                The value to threshold the values in Goggle's LearnedGraph tensor at. Defaults to 0.1.
            decoder_arch: str = "gcn"
                The choice of decoder architecture. Available options are "gcn", "het", and "sage". Defaults to "gcn".
            graph_prior: np.ndarray = None
                The graph_prior used to calculate the loss. Defaults to None.
            prior_mask: np.ndarray = None
                A mask that is applied to the LearnedGraph and graph prior. Defaults to None.
            device: Union[str, torch.device] = synthcity.utils.constants.DEVICE
                The device that the model is run on. Defaults to "cuda" if cuda is available else "cpu".
            alpha: float = 0.1
                The weight applied to the MSE loss in the loss function. Defaults to 0.1.
            beta: float = 0.1
                The weight applied to the loss_graph in the loss function. Defaults to 0.1.
            iter_opt: bool = True
                A flag for optimizing the graph and autoencoder parameters separately. Defaults to True.
            learning_rate: float = 5e-3
                Generator learning rate, used by the Adam optimizer. Defaults to 5e-3.
            weight_decay: float = 1e-3
                Generator weight decay, used by the Adam optimizer. Defaults to 1e-3.
            batch_size: int = 32
                batch size. Defaults to 32.
            patience: int = 50
                Max number of iterations without any improvement before early stopping is triggered. Defaults to 50.
            dataloader_sampler: Any = None
                Optional sampler for the dataloader. Defaults to None.
            logging_epoch: int = 100
                The number of epochs after which information is sent to the debugging level of the logger. Defaults to 100.
            encoder_nonlin: str = "relu"
                The non-linear activation function applied in goggle's encoder. Defaults to "relu".
            decoder_nonlin: str = "relu"
                 The non-linear activation function applied in goggle's decoder. Defaults to "relu".
            encoder_max_clusters: int = 20
                The max number of clusters to create for continuous columns when encoding with TabularEncoder. Defaults to 20.
            encoder_whitelist: list = []
                Ignore columns from encoding with TabularEncoder. Defaults to [].
            decoder_nonlin_out_discrete: str = "softmax"
                Activation function for discrete columns. Useful with the TabularEncoder. Defaults to "softmax".
            decoder_nonlin_out_continuous: str = "tanh
                Activation function for continuous columns. Useful with the TabularEncoder.. Defaults to "tanh".
            random_state: int = 0
                random_state used. Defaults to 0.
        """
        super(TabularGoggle, self).__init__()
        self.columns = X.columns
        self.encoder = TabularEncoder(
            max_clusters=encoder_max_clusters, whitelist=encoder_whitelist
        ).fit(X)

        self.n_iter = n_iter
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.logging_epoch = logging_epoch
        self.batch_size = batch_size
        self.dataloader_sampler = dataloader_sampler
        self.iter_opt = iter_opt
        self.optimiser_gl = None
        self.optimiser_ga = None
        self.optimiser = None
        self.random_state = random_state

        graph_prior = self._check_tensor(graph_prior)
        prior_mask = self._check_tensor(prior_mask)

        self.loss = GoggleLoss(alpha, beta, graph_prior, self.device)
        self.model = Goggle(
            self.encoder.n_features(),
            n_iter,
            encoder_dim,
            encoder_l,
            het_encoding,
            decoder_dim,
            decoder_l,
            threshold,
            decoder_arch,
            graph_prior,
            prior_mask,
            batch_size,
            decoder_nonlin=decoder_nonlin,
            decoder_nonlin_out=self.encoder.activation_layout(
                discrete_activation=decoder_nonlin_out_discrete,
                continuous_activation=decoder_nonlin_out_continuous,
            ),
            encoder_nonlin=encoder_nonlin,
            loss=self.loss,
            iter_opt=self.iter_opt,
            dataloader_sampler=self.dataloader_sampler,
            logging_epoch=self.logging_epoch,
            patience=self.patience,
            random_state=self.random_state,
            device=self.device,
        ).to(device)

        if iter_opt:
            gl_params = ["learned_graph.graph"]
            graph_learner_params = list(
                filter(lambda kv: kv[0] in gl_params, self.model.named_parameters())
            )
            graph_autoencoder_params = list(
                filter(lambda kv: kv[0] not in gl_params, self.model.named_parameters())
            )
            self.optimiser_gl = torch.optim.Adam(
                [param[1] for param in graph_learner_params],
                lr=self.learning_rate,
                weight_decay=0,
            )
            self.optimiser_ga = torch.optim.Adam(
                [param[1] for param in graph_autoencoder_params],
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimiser = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def encode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.transform(X)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def decode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.inverse_transform(X)

    def get_encoder(self) -> TabularEncoder:
        return self.encoder

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        elif X is not None:
            return torch.from_numpy(np.asarray(X)).to(self.device)
        else:
            return X

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        X: pd.DataFrame,
        encoded: bool = False,
        **kwargs: Any,
    ) -> None:
        X_enc = self.encode(X)
        self.model.fit(
            X_enc,
            optimiser_gl=self.optimiser_gl,
            optimiser_ga=self.optimiser_ga,
            optimiser=self.optimiser,
            **kwargs,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate(
        self,
        count: int,
        **kwargs: Any,
    ) -> pd.DataFrame:
        samples = self.forward(count)
        samples = self.decode(pd.DataFrame(samples))
        return pd.DataFrame(samples)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(
        self,
        count: int,
    ) -> torch.Tensor:
        return self.model.generate(count)
