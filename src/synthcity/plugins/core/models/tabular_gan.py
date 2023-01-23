# stdlib
from typing import Any, Callable, Optional, Union

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.preprocessing import OneHotEncoder

# synthcity absolute
from synthcity.metrics.weighted_metrics import WeightedMetrics
from synthcity.utils.constants import DEVICE
from synthcity.utils.samplers import BaseSampler, ConditionalDatasetSampler

# synthcity relative
from .gan import GAN
from .tabular_encoder import TabularEncoder


class TabularGAN(torch.nn.Module):
    """
    .. inheritance-diagram:: synthcity.plugins.core.models.tabular_gan.TabularGAN
        :parts: 1


    GAN for tabular data.

    This class combines GAN and tabular encoder to form a generative model for tabular data.

    Args:
        X: pd.DataFrame
            Reference dataset, used for training the tabular encoder
        n_units_latent: int
            Number of latent units
        cond: Optional
            Optional conditional
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
        generator_nonlin_out: Optional[List[Tuple[str, int]]]
            List of activations. Useful with the TabularEncoder
        generator_lr: float = 2e-4
            Generator learning rate, used by the Adam optimizer
        generator_weight_decay: float = 1e-3
            Generator weight decay, used by the Adam optimizer
        generator_opt_betas: tuple = (0.9, 0.999)
            Generator initial decay rates, used by the Adam Optimizer
        generator_extra_penalties: list
            Additional penalties for the generator. Values: "identifiability_penalty"
        generator_extra_penalty_cbks: List[Callable]
            Additional loss callabacks for the generator. Used by the TabularGAN for the conditional loss
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
        discriminator_lr: float
            Discriminator learning rate, used by the Adam optimizer
        discriminator_weight_decay: float
            Discriminator weight decay, used by the Adam optimizer
        discriminator_opt_betas: tuple
            Initial weight decays for the Adam optimizer
        batch_size: int
            Batch size
        n_iter_print: int
            Number of iterations after which to print updates and check the validation loss.
        random_state: int
            random_state used
        n_iter_min: int
            Minimum number of iterations to go through before starting early stopping
        clipping_value: int, default 0
            Gradients clipping value. Zero disables the feature
        lambda_gradient_penalty: float = 10
            Weight for the gradient penalty
        lambda_identifiability_penalty: float = 0.1
            Weight for the identifiability penalty, if enabled
        dataloader_sampler: Optional[sampler.Sampler]
            Optional sampler for the dataloader, useful for conditional sampling
        device: Any = DEVICE
            CUDA/CPU
        adjust_inference_sampling: bool
            Adjust the marginal probabilities in the synthetic data to closer match the training set. Active only with the ConditionalSampler
        # privacy settings
        dp_enabled: bool
            Train the discriminator with Differential Privacy guarantees
        dp_delta: Optional[float]
            Optional DP delta: the probability of information accidentally being leaked. Usually 1 / len(dataset)
        dp_epsilon: float = 3
            DP epsilon: privacy budget, which is a measure of the amount of privacy that is preserved by a given algorithm. Epsilon is a number that represents the maximum amount of information that an adversary can learn about an individual from the output of a differentially private algorithm. The smaller the value of epsilon, the more private the algorithm is. For example, an algorithm with an epsilon of 0.1 preserves more privacy than an algorithm with an epsilon of 1.0.
        dp_max_grad_norm: float
            max grad norm used for gradient clipping
        dp_secure_mode: bool = False,
             if True uses noise generation approach robust to floating point arithmetic attacks.

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
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 150,
        generator_nonlin: str = "leaky_relu",
        generator_nonlin_out_discrete: str = "softmax",
        generator_nonlin_out_continuous: str = "none",
        generator_n_iter: int = 1000,
        generator_batch_norm: bool = False,
        generator_dropout: float = 0.01,
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
        discriminator_lr: float = 1e-3,
        discriminator_weight_decay: float = 1e-3,
        discriminator_opt_betas: tuple = (0.9, 0.999),
        batch_size: int = 64,
        random_state: int = 0,
        clipping_value: int = 0,
        lambda_gradient_penalty: float = 10,
        lambda_identifiability_penalty: float = 0.1,
        encoder_max_clusters: int = 20,
        encoder: Any = None,
        encoder_whitelist: list = [],
        dataloader_sampler: Optional[BaseSampler] = None,
        device: Any = DEVICE,
        patience: int = 10,
        patience_metric: Optional[WeightedMetrics] = None,
        n_iter_print: int = 50,
        n_iter_min: int = 100,
        adjust_inference_sampling: bool = False,
        # privacy settings
        dp_enabled: bool = False,
        dp_epsilon: float = 3,
        dp_delta: Optional[float] = None,
        dp_max_grad_norm: float = 2,
        dp_secure_mode: bool = False,
    ) -> None:
        super(TabularGAN, self).__init__()
        self.columns = X.columns
        self.batch_size = batch_size
        self.sample_prob: Optional[np.ndarray] = None
        self._adjust_inference_sampling = adjust_inference_sampling
        n_units_conditional = 0

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = TabularEncoder(
                max_clusters=encoder_max_clusters, whitelist=encoder_whitelist
            ).fit(X)

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

        def _generator_cond_loss(
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
                        f"Invalid samples after softmax = {fake_samples[mask, idx : idx + length]}"
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
                raise ValueError(
                    f"Invalid offset idx = {idx}; real_samples.shape = {real_samples.shape}"
                )

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
            discriminator_lr=discriminator_lr,
            discriminator_weight_decay=discriminator_weight_decay,
            discriminator_opt_betas=discriminator_opt_betas,
            lambda_gradient_penalty=lambda_gradient_penalty,
            lambda_identifiability_penalty=lambda_identifiability_penalty,
            clipping_value=clipping_value,
            n_iter_print=n_iter_print,
            random_state=random_state,
            # early stopping
            n_iter_min=n_iter_min,
            dataloader_sampler=dataloader_sampler,
            device=device,
            patience=patience,
            patience_metric=patience_metric,
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

    def get_encoder(self) -> TabularEncoder:
        return self.encoder

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        X: pd.DataFrame,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        fake_labels_generator: Optional[Callable] = None,
        true_labels_generator: Optional[Callable] = None,
        encoded: bool = False,
    ) -> Any:
        # preprocessing
        if encoded:
            X_enc = X
        else:
            X_enc = self.encode(X)

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

        # training
        self.model.fit(
            np.asarray(X_enc),
            np.asarray(cond),
            fake_labels_generator=fake_labels_generator,
            true_labels_generator=true_labels_generator,
        )

        # post processing
        self.adjust_inference_sampling(self._adjust_inference_sampling)

        return self

    def adjust_inference_sampling(self, enabled: bool) -> None:
        if self.predefined_conditional or self.dataloader_sampler is None:
            return

        self._adjust_inference_sampling = enabled

        if enabled:
            real_prob = self.dataloader_sampler.conditional_probs()
            sample_prob = self._extract_sample_prob()

            self.sample_prob = self._find_sample_p(real_prob, sample_prob)
        else:
            self.sample_prob = None

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
        if cond is not None and self.cond_encoder is not None:
            cond = np.asarray(cond)
            if len(cond.shape) == 1:
                cond = cond.reshape(-1, 1)

            cond = self.cond_encoder.transform(cond).toarray()

        if not self.predefined_conditional and self.dataloader_sampler is not None:
            cond = self.dataloader_sampler.sample_conditional(count, p=self.sample_prob)

        return self.model.generate(count, cond=cond)

    def _extract_sample_prob(self) -> Optional[np.ndarray]:
        if self.predefined_conditional or self.dataloader_sampler is None:
            return None

        if self.dataloader_sampler.conditional_dimension() == 0:
            return None

        prob_list = list()
        batch_size = 10000

        for c in range(self.dataloader_sampler.conditional_dimension()):
            cond = self.dataloader_sampler.sample_conditional_for_class(batch_size, c)
            if cond is None:
                continue

            data_cond = self.model.generate(batch_size, cond=cond)

            syn_dataloader_sampler = ConditionalDatasetSampler(
                pd.DataFrame(data_cond),
                self.encoder.layout(),
            )

            prob = syn_dataloader_sampler.conditional_probs()
            prob_list.append(prob)

        prob_mat = np.stack(prob_list, axis=-1)

        return prob_mat

    def _find_sample_p(
        self, prob_real: Optional[np.ndarray], prob_mat: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if prob_real is None or prob_mat is None:
            return None

        def kl(
            alpha: np.ndarray, prob_real: np.ndarray, prob_mat: np.ndarray
        ) -> np.ndarray:
            # alpha: _n_categories

            # f1: same as prob_real
            alpha_tensor = alpha[None, None, :]
            f1 = logsumexp(alpha_tensor, axis=-1, b=prob_mat)
            f2 = logsumexp(alpha)
            ce = -np.sum(prob_real * f1, axis=1) + f2
            return np.mean(ce)

        try:
            res = minimize(kl, np.ones(prob_mat.shape[-1]), (prob_real, prob_mat))
        except Exception:
            return np.ones(prob_mat.shape[-1]) / prob_mat.shape[-1]

        return np.exp(res.x) / np.sum(np.exp(res.x))
