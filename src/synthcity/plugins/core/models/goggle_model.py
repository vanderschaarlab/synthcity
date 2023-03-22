# stdlib
from typing import Any

# third party
import pandas as pd
import torch
from pydantic import validate_arguments

# synthcity absolute
from synthcity.utils.constants import DEVICE
from synthcity.utils.samplers import ConditionalDatasetSampler

# synthcity relative
from .goggle import Goggle, GoggleLoss
from .tabular_encoder import TabularEncoder


class GoggleModel:
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
        graph_prior: Any = None,  # torch.Tensor
        prior_mask: Any = None,  # torch.Tensor
        device: str = DEVICE,
        alpha: float = 0.1,
        beta: float = 0.1,
        iter_opt: bool = True,
        learning_rate: float = 5e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 32,
        patience: int = 50,
        dataloader_sampler: Any = None,
        logging_epoch: int = 100,
        tabular_encoder: Any = None,
        encoder_max_clusters: int = 20,
        encoder_whitelist: list = [],
    ):
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

        if tabular_encoder is not None:
            self.tabular_encoder = tabular_encoder
        else:
            self.tabular_encoder = TabularEncoder(
                max_clusters=encoder_max_clusters, whitelist=encoder_whitelist
            ).fit(X)

        self.loss = GoggleLoss(alpha, beta, graph_prior, device)
        self.model = Goggle(
            self.encode(X).shape[1],
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
            loss=self.loss,
            learning_rate=learning_rate,
            iter_opt=self.iter_opt,
            dataloader_sampler=self.dataloader_sampler,
            logging_epoch=self.logging_epoch,
            patience=self.patience,
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

        if dataloader_sampler is None:
            dataloader_sampler = ConditionalDatasetSampler(
                self.tabular_encoder.transform(X),
                self.tabular_encoder.layout(),
            )

        self.dataloader_sampler = dataloader_sampler

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def encode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.tabular_encoder.transform(X)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def decode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.tabular_encoder.inverse_transform(X)

    def get_encoder(self) -> TabularEncoder:
        return self.tabular_encoder

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        X: pd.DataFrame,
        encoded: bool = False,
        **kwargs: Any,
    ) -> None:
        # preprocessing
        if encoded:
            X_enc = X
        else:
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
        samples = self.forward(count)  # , cond)
        return self.decode(pd.DataFrame(samples))

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(
        self,
        count: int,
    ) -> torch.Tensor:
        return self.model.generate(count)
