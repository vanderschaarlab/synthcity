# stdlib
from typing import Any, Callable, Optional, Union

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments

# synthcity absolute
from synthcity.utils.constants import DEVICE
from synthcity.utils.samplers import BaseSampler, ConditionalDatasetSampler

# synthcity relative
from .gan import GAN
from .tabular_encoder import TabularEncoder


class TabularGAN(torch.nn.Module):
    """
    GAN for tabular data.

    This class combines GAN and tabular encoder to form a generative model for tabular data.

    Args:
        n_units_in: int
            Number of features
        generator_n_layers_hidden: int
            Number of hidden layers in the generator
        generator_n_units_hidden: int
            Number of hidden units in each layer of the Generator
        generator_nonlin: string, default 'elu'
            Nonlinearity to use in the generator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        generator_n_iter: int
            Maximum number of iterations in the Generator.
        generator_batch_norm: bool
            Enable/disable batch norm for the generator
        generator_dropout: float
            Dropout value. If 0, the dropout is not used.
        generator_residual: bool
            Use residuals for the generator
        discriminator_n_layers_hidden: int
            Number of hidden layers in the discriminator
        discriminator_n_units_hidden: int
            Number of hidden units in each layer of the discriminator
        discriminator_nonlin: string, default 'relu'
            Nonlinearity to use in the discriminator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        discriminator_n_iter: int
            Maximum number of iterations in the discriminator.
        discriminator_batch_norm: bool
            Enable/disable batch norm for the discriminator
        discriminator_dropout: float
            Dropout value for the discriminator. If 0, the dropout is not used.
        lr: float
            learning rate for optimizer. step_size equivalent in the JAX version.
        weight_decay: float
            l2 (ridge) penalty for the weights.
        batch_size: int
            Batch size
        n_iter_print: int
            Number of iterations after which to print updates and check the validation loss.
        random_state: int
            random_state used
        n_iter_min: int
            Minimum number of iterations to go through before starting early stopping
        clipping_value: int, default 0
            Gradients clipping value
        lambda_gradient_penalty: float
            Lambda weight for the gradient penalty
        lambda_identifiability_penalty: float
            Lambda weight for the identifiability loss
        encoder_max_clusters: int
            The max number of clusters to create for continuous columns when encoding
        encoder:
            Pre-trained tabular encoder. If None, a new encoder is trained.
        encoder_whitelist:
            Ignore columns from encoding
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        X: pd.DataFrame,
        n_units_latent: int,
        n_units_conditional: int = 0,
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 150,
        generator_nonlin: str = "leaky_relu",
        generator_nonlin_out_discrete: str = "softmax",
        generator_nonlin_out_continuous: str = "none",
        generator_n_iter: int = 1000,
        generator_batch_norm: bool = False,
        generator_dropout: float = 0.01,
        generator_loss: Optional[Callable] = None,
        generator_lr: float = 1e-3,
        generator_weight_decay: float = 1e-3,
        generator_opt_betas: tuple = (0.9, 0.999),
        generator_residual: bool = True,
        generator_extra_penalties: list = [],  # "identifiability_penalty"
        discriminator_n_layers_hidden: int = 3,
        discriminator_n_units_hidden: int = 300,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_batch_norm: bool = False,
        discriminator_dropout: float = 0.1,
        discriminator_loss: Optional[Callable] = None,
        discriminator_lr: float = 1e-3,
        discriminator_weight_decay: float = 1e-3,
        discriminator_opt_betas: tuple = (0.9, 0.999),
        batch_size: int = 64,
        n_iter_print: int = 100,
        random_state: int = 0,
        n_iter_min: int = 100,
        clipping_value: int = 0,
        lambda_gradient_penalty: float = 10,
        lambda_identifiability_penalty: float = 0.1,
        encoder_max_clusters: int = 20,
        encoder: Any = None,
        encoder_whitelist: list = [],
        dataloader_sampler: Optional[BaseSampler] = None,
        device: Any = DEVICE,
        # privacy settings
        dp_enabled: bool = False,
        dp_epsilon: float = 3,
        dp_delta: Optional[float] = None,
        dp_max_grad_norm: float = 2,
        dp_secure_mode: bool = False,
    ) -> None:
        super(TabularGAN, self).__init__()
        self.columns = X.columns
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = TabularEncoder(
                max_clusters=encoder_max_clusters, whitelist=encoder_whitelist
            ).fit(X)

        if dataloader_sampler is None:
            dataloader_sampler = ConditionalDatasetSampler(
                self.encoder.transform(X),
                self.encoder.layout(),
            )
            n_units_conditional += dataloader_sampler.conditional_dimension()

        self.dataloader_sampler = dataloader_sampler

        def _generator_cond_loss(
            real_samples: torch.tensor,
            fake_samples: torch.Tensor,
            cond: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if cond is None:
                return 0

            losses = []

            idx = 0
            cond_idx = 0

            for item in self.encoder.layout():
                length = item.output_dimensions

                if item.column_type != "discrete":
                    idx += length
                    continue

                # create activate feature mask
                mask = cond[:, cond_idx : cond_idx + length].sum(axis=1).bool()

                if mask.sum() == 0:
                    idx += length
                    continue

                assert (
                    fake_samples[mask, idx : idx + length] >= 0
                ).all(), fake_samples[mask, idx : idx + length]
                # fake_samples are after the Softmax activation
                # we filter active features in the mask
                item_loss = torch.nn.NLLLoss()(
                    torch.log(fake_samples[mask, idx : idx + length] + 1e-8),
                    torch.argmax(real_samples[mask, idx : idx + length], dim=1),
                )
                losses.append(item_loss)

                cond_idx += length
                idx += length

            assert idx == real_samples.shape[1]

            if len(losses) == 0:
                return 0

            loss = torch.stack(losses, dim=-1)

            return loss.sum() / len(real_samples)

        self.model = GAN(
            self.encoder.n_features(),
            n_units_latent=n_units_latent,
            n_units_conditional=n_units_conditional,
            batch_size=batch_size,
            generator_n_layers_hidden=generator_n_layers_hidden,
            generator_n_units_hidden=generator_n_units_hidden,
            generator_nonlin=generator_nonlin,
            generator_nonlin_out=self.encoder.activation_layout(
                discrete_activation=generator_nonlin_out_discrete,
                continuous_activation=generator_nonlin_out_continuous,
            ),
            generator_n_iter=generator_n_iter,
            generator_batch_norm=generator_batch_norm,
            generator_dropout=generator_dropout,
            generator_loss=generator_loss,
            generator_lr=generator_lr,
            generator_residual=generator_residual,
            generator_weight_decay=generator_weight_decay,
            generator_opt_betas=generator_opt_betas,
            generator_extra_penalties=generator_extra_penalties,
            generator_extra_penalty_cbks=[_generator_cond_loss],
            discriminator_n_units_hidden=discriminator_n_units_hidden,
            discriminator_n_layers_hidden=discriminator_n_layers_hidden,
            discriminator_n_iter=discriminator_n_iter,
            discriminator_nonlin=discriminator_nonlin,
            discriminator_batch_norm=discriminator_batch_norm,
            discriminator_dropout=discriminator_dropout,
            discriminator_loss=discriminator_loss,
            discriminator_lr=discriminator_lr,
            discriminator_weight_decay=discriminator_weight_decay,
            discriminator_opt_betas=discriminator_opt_betas,
            lambda_gradient_penalty=lambda_gradient_penalty,
            lambda_identifiability_penalty=lambda_identifiability_penalty,
            clipping_value=clipping_value,
            n_iter_print=n_iter_print,
            random_state=random_state,
            n_iter_min=n_iter_min,
            dataloader_sampler=dataloader_sampler,
            device=device,
            # privacy
            dp_enabled=dp_enabled,
            dp_epsilon=dp_epsilon,
            dp_delta=dp_delta,
            dp_max_grad_norm=dp_max_grad_norm,
            dp_secure_mode=dp_secure_mode,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def encode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.transform(X)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def decode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.inverse_transform(X)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        X: pd.DataFrame,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        fake_labels_generator: Optional[Callable] = None,
        true_labels_generator: Optional[Callable] = None,
        encoded: bool = False,
    ) -> Any:
        if encoded:
            X_enc = X
        else:
            X_enc = self.encode(X)

        extra_cond = self.dataloader_sampler.get_train_conditionals()
        cond = self._merge_conditionals(cond, extra_cond)
        self.model.fit(
            np.asarray(X_enc),
            np.asarray(cond),
            fake_labels_generator=fake_labels_generator,
            true_labels_generator=true_labels_generator,
        )
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate(
        self,
        count: int,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    ) -> pd.DataFrame:
        samples = self(count, cond)
        return self.decode(pd.DataFrame(samples))

    def forward(
        self, count: int, cond: Optional[Union[pd.DataFrame, np.ndarray]] = None
    ) -> torch.Tensor:
        extra_cond = self.dataloader_sampler.sample_conditional(count)
        cond = self._merge_conditionals(cond, extra_cond)

        return self.model.generate(count, cond=cond)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _merge_conditionals(
        self,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]],
        extra_cond: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if extra_cond is None and cond is None:
            return None

        if extra_cond is None:
            return cond

        if cond is None:
            cond = extra_cond
        else:
            cond = np.concatenate([extra_cond, np.asarray(cond)], axis=1)

        return cond
