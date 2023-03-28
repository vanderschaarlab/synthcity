# stdlib
from typing import Any, Optional

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
        schema: Any = None,
        encoder_nonlin: str = "leaky_relu",
        decoder_nonlin: str = "leaky_relu",
        encoder_max_clusters: int = 20,
        encoder_whitelist: list = [],
        decoder_nonlin_out_discrete: str = "softmax",
        decoder_nonlin_out_continuous: str = "tanh",
    ):
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
        self.schema = schema

        graph_prior = self._check_tensor(graph_prior)
        prior_mask = self._check_tensor(prior_mask)

        self.loss = GoggleLoss(alpha, beta, graph_prior, device)
        self.model = Goggle(
            self.encoder.n_features(),  # X.shape[1],
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

    def enforce_constraints(self, X_synth: torch.Tensor) -> np.ndarray:
        X_synth = pd.DataFrame(X_synth, columns=self.schema.features())
        for rule in self.schema.as_constraints().rules:
            if rule[1] == "in":
                X_synth[rule[0]] = X_synth[rule[0]].apply(
                    lambda x: min(rule[2], key=lambda z: abs(z - x))
                )
            elif rule[1] == "eq":
                raise Exception("not yet implemented")
            else:
                pass
        return X_synth.values

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate(
        self,
        count: int,
        **kwargs: Any,
    ) -> pd.DataFrame:
        samples = self.forward(count)  # , cond)
        samples = self.enforce_constraints(self.decode(pd.DataFrame(samples)))
        return pd.DataFrame(samples, columns=self.schema)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(
        self,
        count: int,
    ) -> torch.Tensor:
        return self.model.generate(count)
