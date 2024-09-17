# stdlib
from typing import Any, Optional, Union

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments
from sklearn.preprocessing import OneHotEncoder
from torch import nn

# synthcity absolute
from synthcity.utils.constants import DEVICE
from synthcity.utils.samplers import BaseSampler, ConditionalDatasetSampler
from synthcity.plugins.core.dataloader import DataLoader

# synthcity relative
from .tabular_encoder import TabularEncoder
from .vae import VAE
from .fasd import FASD, FASD_Decoder


class TabularFASD(nn.Module):
    """
    .. inheritance-diagram:: synthcity.plugins.core.models.tabular_vae.TabularVAE
        :parts: 1


    VAE for tabular data.

    This class combines VAE and tabular encoder to form a generative model for tabular data.

    Args:
        X: pd.DataFrame
            Reference dataset, used for training the tabular encoder
        cond: Optional
            Optional conditional
        decoder_n_layers_hidden: int
            Number of hidden layers in the decoder
        decoder_n_units_hidden: int
            Number of hidden units in each layer of the decoder
        decoder_nonlin: string, default 'elu'
            Nonlinearity to use in the decoder. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        decoder_n_iter: int
            Maximum number of iterations in the decoder.
        decoder_batch_norm: bool
            Enable/disable batch norm for the decoder
        decoder_dropout: float
            Dropout value. If 0, the dropout is not used.
        decoder_residual: bool
            Use residuals for the decoder
        encoder_n_layers_hidden: int
            Number of hidden layers in the encoder
        encoder_n_units_hidden: int
            Number of hidden units in each layer of the encoder
        encoder_nonlin: string, default 'relu'
            Nonlinearity to use in the encoder. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        encoder_n_iter: int
            Maximum number of iterations in the encoder.
        encoder_batch_norm: bool
            Enable/disable batch norm for the encoder
        encoder_dropout: float
            Dropout value for the encoder. If 0, the dropout is not used.
        lr: float
            learning rate for optimizer.
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
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        X_gt: DataLoader,
        n_units_embedding: int,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        lr: float = 2e-4,
        n_iter: int = 500,
        weight_decay: float = 1e-3,
        batch_size: int = 64,
        random_state: int = 0,
        loss_strategy: str = "standard",
        encoder_max_clusters: int = 20,
        decoder_n_layers_hidden: int = 2,
        decoder_n_units_hidden: int = 250,
        decoder_nonlin: str = "leaky_relu",
        decoder_nonlin_out_discrete: str = "softmax",
        decoder_nonlin_out_continuous: str = "tanh",
        decoder_batch_norm: bool = False,
        decoder_dropout: float = 0,
        decoder_residual: bool = True,
        encoder_n_layers_hidden: int = 3,
        encoder_n_units_hidden: int = 300,
        encoder_nonlin: str = "leaky_relu",
        encoder_batch_norm: bool = False,
        encoder_dropout: float = 0.1,
        fasd_n_iter: int = 300,
        fasd_n_units_embedding: int = 100,
        fasd_encoder_n_layers_hidden: int = 0,
        fasd_encoder_n_units_hidden: int = 0,
        fasd_encoder_nonlin: str = "none",
        fasd_encoder_dropout: float = 0.1,
        fasd_decoder_n_layers_hidden: int = 0,
        fasd_decoder_n_units_hidden: int = 0,
        fasd_decoder_nonlin: str = "none",
        fasd_decoder_dropout: float = 0.1,
        encoder_whitelist: list = [],
        device: Any = DEVICE,
        robust_divergence_beta: int = 2,  # used for loss_strategy = robust_divergence
        loss_factor: int = 1,  # used for standar losss
        dataloader_sampler: Optional[BaseSampler] = None,
        clipping_value: int = 1,
        # early stopping
        n_iter_min: int = 100,
        n_iter_print: int = 10,
        patience: int = 20,
    ) -> None:
        super(TabularFASD, self).__init__()
        X = X_gt.dataframe()
        self.target_column = X_gt.target_column

        # separately encode X and y
        self.data_encoder = TabularEncoder(
            continuous_encoder="minmax", cont_encoder_params={"feature_range": (-1, 1)}
        )
        self.data_encoder.fit(
            X.drop(self.target_column, axis=1), discrete_columns=X_gt.discrete_features
        )
        X_enc = self.data_encoder.transform(X.drop(self.target_column, axis=1))
        self.target_encoder = TabularEncoder(
            continuous_encoder="minmax", cont_encoder_params={"feature_range": (-1, 1)}
        )
        self.target_encoder.fit((X[self.target_column]).to_frame())
        y_enc = self.target_encoder.transform((X[self.target_column]).to_frame())
        # train the FASD model to get representations
        self.fasd_model = FASD(
            n_units_in=self.data_encoder.n_features(),
            n_units_embedding=fasd_n_units_embedding,
            n_units_hidden=fasd_encoder_n_units_hidden,
            n_layers_hidden=fasd_encoder_n_layers_hidden,
            hidden_nonlin=fasd_encoder_nonlin,
            target_nonlin_out=self.target_encoder.activation_layout(
                discrete_activation="softmax", continuous_activation="tanh"
            ),
            device=device,
            random_state=random_state,
            batch_size=batch_size,
            n_iter=fasd_n_iter,
            patience=50,
            lr=lr,
            weight_decay=weight_decay,
            n_iter_min=30,
            dropout=fasd_encoder_dropout,
            clipping_value=clipping_value,
        )
        self.fasd_model.fit(X_enc, y_enc)
        X_rep = self.fasd_model.encode(X_enc)

        # standardise representations
        self.encoder = TabularEncoder(
            continuous_encoder="standard",
            cont_encoder_params={},
            categorical_encoder="passthrough",
            cat_encoder_params={},
            categorical_limit=2,
        )
        self.encoder.fit(X_rep)
        X_rep = self.encoder.transform(X_rep)

        # train the decoder (standardised representations to original input features)
        self.fasd_decoder = FASD_Decoder(
            n_units_in=self.encoder.n_features(),
            n_units_hidden=fasd_decoder_n_units_hidden,  # shallow decoder
            n_units_out=self.data_encoder.n_features(),
            n_layers_hidden=fasd_decoder_n_layers_hidden,
            nonlin=fasd_decoder_nonlin,
            nonlin_out=self.data_encoder.activation_layout(
                discrete_activation="softmax", continuous_activation="tanh"
            ),
            device=device,
            random_state=random_state,
            batch_size=batch_size,
            n_iter=fasd_n_iter,
            patience=50,
            lr=lr,
            weight_decay=weight_decay,
            n_iter_min=30,
            dropout=fasd_decoder_dropout,
            clipping_value=clipping_value,
        )
        self.fasd_decoder.fit(X_rep, X_enc)

        # set output activation of VAE to none (standard scaled data)
        decoder_nonlin_out_continuous = "none"

        # set raw data as representations for conditionals
        X = X_rep.copy()

        # store copy of representations for the fit method
        self.X_rep = X_rep.copy()

        n_units_conditional = 0
        self.cond_encoder: Optional[OneHotEncoder] = None
        if cond is not None:
            cond = np.asarray(cond)
            if len(cond.shape) == 1:
                cond = cond.reshape(-1, 1)

            self.cond_encoder = OneHotEncoder(handle_unknown="ignore").fit(cond)
            cond = self.cond_encoder.transform(cond).toarray()

            n_units_conditional = cond.shape[-1]

        self.predefined_conditional = cond is not None

        if (
            dataloader_sampler is None and not self.predefined_conditional
        ):  # don't mix conditionals
            dataloader_sampler = ConditionalDatasetSampler(
                self.encoder.transform(X),
                self.encoder.layout(),
            )
            n_units_conditional = dataloader_sampler.conditional_dimension()

        self.dataloader_sampler = dataloader_sampler

        def _cond_loss(
            real_samples: torch.tensor,
            fake_samples: torch.Tensor,
            cond: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if cond is None or self.predefined_conditional:
                return 0

            losses = []

            idx = 0
            cond_idx = 0

            for item in self.encoder.layout():
                length = item.output_dimensions

                if item.feature_type != "discrete":
                    idx += length
                    continue

                # create activate feature mask
                mask = cond[:, cond_idx : cond_idx + length].sum(axis=1).bool()

                if mask.sum() == 0:
                    idx += length
                    continue

                if not (fake_samples[mask, idx : idx + length] >= 0).all():
                    raise RuntimeError(
                        f"Values should be positive after softmax = {fake_samples[mask, idx : idx + length]}"
                    )
                # fake_samples are after the Softmax activation
                # we filter active features in the mask
                item_loss = torch.nn.NLLLoss()(
                    torch.log(fake_samples[mask, idx : idx + length] + 1e-8),
                    torch.argmax(real_samples[mask, idx : idx + length], dim=1),
                )
                losses.append(item_loss)

                cond_idx += length
                idx += length

            if idx != real_samples.shape[1]:
                raise RuntimeError(f"Invalid offset {idx} {real_samples.shape}")

            if len(losses) == 0:
                return 0

            loss = torch.stack(losses, dim=-1)
            return loss.sum() / len(real_samples)

        self.model = VAE(
            self.encoder.n_features(),
            n_units_embedding=n_units_embedding,
            n_units_conditional=n_units_conditional,
            batch_size=batch_size,
            n_iter=n_iter,
            lr=lr,
            weight_decay=weight_decay,
            random_state=random_state,
            loss_strategy=loss_strategy,
            decoder_n_layers_hidden=decoder_n_layers_hidden,
            decoder_n_units_hidden=decoder_n_units_hidden,
            decoder_nonlin=decoder_nonlin,
            decoder_nonlin_out=self.encoder.activation_layout(
                discrete_activation=decoder_nonlin_out_discrete,
                continuous_activation=decoder_nonlin_out_continuous,
            ),
            decoder_batch_norm=decoder_batch_norm,
            decoder_dropout=decoder_dropout,
            decoder_residual=decoder_residual,
            encoder_n_units_hidden=encoder_n_units_hidden,
            encoder_n_layers_hidden=encoder_n_layers_hidden,
            encoder_nonlin=encoder_nonlin,
            encoder_batch_norm=encoder_batch_norm,
            encoder_dropout=encoder_dropout,
            dataloader_sampler=dataloader_sampler,
            device=device,
            extra_loss_cbks=[_cond_loss],
            robust_divergence_beta=robust_divergence_beta,
            loss_factor=loss_factor,
            clipping_value=clipping_value,
            n_iter_print=n_iter_print,
            n_iter_min=n_iter_min,
            patience=patience,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def encode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.transform(X)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def decode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.inverse_transform(X)

    def get_encoder(self) -> TabularEncoder:
        return self.encoder

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Any:
        X_enc = self.X_rep.copy()

        if cond is not None and self.cond_encoder is not None:
            cond = np.asarray(cond)
            if len(cond.shape) == 1:
                cond = cond.reshape(-1, 1)

            cond = self.cond_encoder.transform(cond).toarray()

        if not self.predefined_conditional and self.dataloader_sampler is not None:
            cond = self.dataloader_sampler.get_dataset_conditionals()

        if cond is not None:
            if len(cond) != len(X_enc):
                raise ValueError(
                    f"Invalid conditional shape. {cond.shape} expected {len(X_enc)}"
                )

        self.model.fit(X_enc, cond, **kwargs)
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate(
        self,
        count: int,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        # draw (standardised) representations from the VAE
        samples = pd.DataFrame(self(count, cond))

        # synthesize targets from synthetic representations
        # however, first decode the standardization, since the predictor saw unstandardized data during training
        y = self.fasd_model.predict(self.encoder.inverse_transform(samples))
        # remove target encoding
        y = self.target_encoder.inverse_transform(y)

        # decode (standardised) synthetic representations to original data space
        # we do not have to inverse the standardisation, since decoder was trained on standardised representations already
        samples = self.fasd_decoder.decode(samples)

        # remove tabular encoding of the reconstructed input features
        samples = self.data_encoder.inverse_transform(samples)

        # attach targets to synthetic data
        samples[self.target_column] = y

        return samples

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(
        self,
        count: int,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    ) -> torch.Tensor:
        if cond is not None and self.cond_encoder is not None:
            cond = np.asarray(cond)
            if len(cond.shape) == 1:
                cond = cond.reshape(-1, 1)

            cond = self.cond_encoder.transform(cond).toarray()

        if not self.predefined_conditional and self.dataloader_sampler is not None:
            cond = self.dataloader_sampler.sample_conditional(count)

        return self.model.generate(count, cond=cond)
