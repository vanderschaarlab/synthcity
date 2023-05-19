"""
Based on
- https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
- https://github.com/ehoogeboom/multinomial_diffusion
- https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
# stdlib
import math
from typing import Any, Optional, Tuple

# third party
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# synthcity absolute
from synthcity.logger import debug, info, warning

# synthcity relative
from .modules import DiffusionModel
from .utils import (
    discretized_gaussian_log_likelihood,
    index_to_log_onehot,
    log_1_min_a,
    log_add_exp,
    log_categorical,
    mean_flat,
    normal_kl,
    ohe_to_categories,
    perm_and_expand,
    sliced_logsumexp,
    sum_except_batch,
)


def get_beta_schedule(schedule_name: str, num_diffusion_timesteps: int) -> np.ndarray:
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        # Create a beta schedule that discretizes the given alpha_t_bar function,
        # which defines the cumulative product of (1-beta) over time from t = [0,1].
        def alpha_bar(t: float) -> float:
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

        # a lambda that takes an argument t between 0 and 1 and produces the cumulative
        # product of (1-beta) up to that part of the diffusion process.
        max_beta = 0.999
        # the maximum beta to use; use values lower than 1 to prevent singularities.
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


class GaussianMultinomialDiffusion(torch.nn.Module):
    def __init__(
        self,
        *,
        num_numerical_features: int,
        num_categorical_features: tuple,
        model_type: str,
        model_params: dict,
        num_timesteps: int = 1000,
        num_classes: int = 0,
        conditional: bool = False,
        dim_emb: int = 128,
        gaussian_loss_type: str = "mse",
        gaussian_parametrization: str = "eps",
        multinomial_loss_type: str = "vb_stochastic",
        parametrization: str = "x0",
        scheduler: str = "cosine",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super(GaussianMultinomialDiffusion, self).__init__()
        if multinomial_loss_type not in ("vb_stochastic", "vb_all"):
            raise ValueError(
                "multinomial_loss_type must be 'vb_stochastic' or 'vb_all'"
            )
        if gaussian_loss_type not in ("mse", "kl"):
            raise ValueError("gaussian_loss_type must be 'mse' or 'kl'")
        if parametrization not in ("x0", "direct"):
            raise ValueError("parametrization must be 'x0' or 'direct'")

        if multinomial_loss_type == "vb_all":
            warning(
                "Computing the loss using the bound on _all_ timesteps."
                " This is expensive both in terms of memory and computation."
            )

        self.num_numerics = num_numerical_features
        self.num_classes = np.asarray(num_categorical_features)
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate(
                [np.repeat(k, k) for k in num_categorical_features], dtype=np.float32
            )
        ).to(device)
        self.dim_input = self.num_numerics + sum(self.num_classes)

        self.slices_for_classes = [np.arange(self.num_classes[0])]
        offsets = np.cumsum(self.num_classes)
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(np.append([0], offsets)).to(device).long()

        self.denoise_fn = DiffusionModel(
            dim_in=self.dim_input,
            dim_emb=dim_emb,
            num_classes=num_classes,
            conditional=conditional,
            model_type=model_type,
            model_params=model_params,
        )

        self.gaussian_loss_type = gaussian_loss_type
        self.gaussian_parametrization = gaussian_parametrization
        self.multinomial_loss_type = multinomial_loss_type
        self.num_timesteps = num_timesteps
        self.parametrization = parametrization
        self.scheduler = scheduler

        alphas = 1.0 - get_beta_schedule(scheduler, num_timesteps)
        alphas = torch.tensor(alphas.astype("float64"))
        betas = 1.0 - alphas

        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.tensor(np.append(1.0, alphas_cumprod[:-1]))
        alphas_cumprod_next = torch.tensor(np.append(alphas_cumprod[1:], 0.0))
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        # Gaussian diffusion

        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)

        self.posterior_log_variance_clipped = (
            torch.from_numpy(
                np.log(
                    np.append(
                        self.posterior_variance[1].cpu(),
                        self.posterior_variance[1:].cpu(),
                    )
                )
            )
            .float()
            .to(device)
        )

        self.posterior_mean_coef1 = (
            ((betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
            .float()
            .to(device)
        )

        self.posterior_mean_coef2 = (
            (
                (1.0 - alphas_cumprod_prev)
                * np.sqrt(alphas.numpy())
                / (1.0 - alphas_cumprod)
            )
            .float()
            .to(device)
        )

        if (
            max(
                log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item(),
                log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha)
                .abs()
                .sum()
                .item(),
                (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item(),
            )
            > 1e-5
        ):
            raise ValueError("Numerical error in log-sum-exp")

        # Convert to float32 and register buffers.
        self.register_buffer("alphas", alphas.float().to(device))
        self.register_buffer("log_alpha", log_alpha.float().to(device))
        self.register_buffer("log_1_min_alpha", log_1_min_alpha.float().to(device))
        self.register_buffer(
            "log_1_min_cumprod_alpha", log_1_min_cumprod_alpha.float().to(device)
        )
        self.register_buffer("log_cumprod_alpha", log_cumprod_alpha.float().to(device))
        self.register_buffer("alphas_cumprod", alphas_cumprod.float().to(device))
        self.register_buffer(
            "alphas_cumprod_prev", alphas_cumprod_prev.float().to(device)
        )
        self.register_buffer(
            "alphas_cumprod_next", alphas_cumprod_next.float().to(device)
        )
        self.register_buffer(
            "sqrt_alphas_cumprod", sqrt_alphas_cumprod.float().to(device)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            sqrt_one_minus_alphas_cumprod.float().to(device),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod.float().to(device)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            sqrt_recipm1_alphas_cumprod.float().to(device),
        )

        self.register_buffer("Lt_history", torch.zeros(num_timesteps))
        self.register_buffer("Lt_count", torch.zeros(num_timesteps))

    # Gaussian part
    def gaussian_q_mean_variance(
        self, x_start: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        mean = perm_and_expand(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = perm_and_expand(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = perm_and_expand(self.log_1_min_cumprod_alpha, t, x_start.shape)
        return mean, variance, log_variance

    def gaussian_q_sample(
        self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None
    ) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        if noise.shape != x_start.shape:
            raise ValueError("noise.shape != x_start.shape")
        return (
            perm_and_expand(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + perm_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def gaussian_q_posterior_mean_variance(
        self, x_start: Tensor, x_t: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if x_start.shape != x_t.shape:
            raise ValueError("x_start.shape != x_t.shape")
        posterior_mean = (
            perm_and_expand(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + perm_and_expand(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = perm_and_expand(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = perm_and_expand(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        if not (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        ):
            raise ValueError("tensor lengths mismatch")
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def gaussian_p_mean_variance(
        self,
        model_output: Tensor,
        x: Tensor,
        t: Tensor,
        model_kwargs: Optional[dict] = None,
    ) -> dict:
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        if t.shape != (B,):
            raise ValueError("length of t is not equal to batch size")

        model_variance = torch.cat(
            [
                self.posterior_variance[1].unsqueeze(0),
                (1.0 - self.alphas)[1:],
            ],
            dim=0,
        )
        model_log_variance = torch.log(model_variance)

        model_variance = perm_and_expand(model_variance, t, x.shape)
        model_log_variance = perm_and_expand(model_log_variance, t, x.shape)

        if self.gaussian_parametrization == "eps":
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        elif self.gaussian_parametrization == "x0":
            pred_xstart = model_output
        else:
            raise ValueError("unknown gaussian_parametrization. Must be 'eps' or 'x0'")

        model_mean, _, _ = self.gaussian_q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        if not (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        ):
            raise ValueError(
                "not all of model_mean, model_log_variance, pred_xstart, x have the same shape"
            )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _vb_terms_bpd(
        self,
        model_output: Tensor,
        x_start: Tensor,
        x_t: Tensor,
        t: Tensor,
        model_kwargs: Optional[dict] = None,
    ) -> dict:
        (
            true_mean,
            _,
            true_log_variance_clipped,
        ) = self.gaussian_q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        out = self.gaussian_p_mean_variance(
            model_output, x_t, t, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        if decoder_nll.shape != x_start.shape:
            raise ValueError("decoder_nll.shape != x_start.shape")
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {
            "output": output,
            "pred_xstart": out["pred_xstart"],
            "out_mean": out["mean"],
            "true_mean": true_mean,
        }

    def _prior_gaussian(self, x_start: Tensor) -> Tensor:
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.gaussian_q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def _gaussian_loss(
        self,
        model_out: Tensor,
        x_start: Tensor,
        x_t: Tensor,
        t: Tensor,
        noise: Tensor,
        model_kwargs: Optional[dict] = None,
    ) -> Tensor:
        if model_kwargs is None:
            model_kwargs = {}

        terms = {}
        if self.gaussian_loss_type == "mse":
            terms["loss"] = mean_flat((noise - model_out) ** 2)
        elif self.gaussian_loss_type == "kl":
            terms["loss"] = self._vb_terms_bpd(
                model_output=model_out,
                x_start=x_start,
                x_t=x_t,
                t=t,
                model_kwargs=model_kwargs,
            )["output"]

        return terms["loss"]

    def _predict_xstart_from_eps(
        self, x_t: Tensor, t: Tensor, eps: Tensor = 1e-08
    ) -> Tensor:
        if x_t.shape != eps.shape:
            raise ValueError("x_t.shape != eps.shape")
        return (
            perm_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - perm_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(
        self, x_t: Tensor, t: Tensor, pred_xstart: Tensor
    ) -> Tensor:
        return (
            perm_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / perm_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def gaussian_p_sample(
        self,
        model_out: Tensor,
        x: Tensor,
        t: Tensor,
        model_kwargs: Optional[dict] = None,
    ) -> dict:
        out = self.gaussian_p_mean_variance(
            model_out,
            x,
            t,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        sample = (
            out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        )
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    # Multinomial part

    def multinomial_kl(self, log_prob1: Tensor, log_prob2: Tensor) -> Tensor:
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t: Tensor, t: Tensor) -> Tensor:
        log_alpha_t = perm_and_expand(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = perm_and_expand(self.log_1_min_alpha, t, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - torch.log(self.num_classes_expanded),
        )

        return log_probs

    def q_pred(self, log_x_start: Tensor, t: Tensor) -> Tensor:
        log_cumprod_alpha_t = perm_and_expand(
            self.log_cumprod_alpha, t, log_x_start.shape
        )
        log_1_min_cumprod_alpha = perm_and_expand(
            self.log_1_min_cumprod_alpha, t, log_x_start.shape
        )

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - torch.log(self.num_classes_expanded),
        )

        return log_probs

    def predict_start(self, model_out: Tensor, log_x_t: Tensor) -> Tensor:
        if model_out.size(0) != log_x_t.size(0):
            raise ValueError(
                f"length of model_out {model_out.size(0)} != length of log_x_t {log_x_t.size(0)}"
            )
        if model_out.size(1) != self.num_classes.sum():
            raise ValueError(
                f"length of model_out {model_out.size(1)} != total num_classes {self.num_classes.sum()}"
            )

        log_pred = torch.empty_like(model_out)
        for ix in self.slices_for_classes:
            log_pred[:, ix] = F.log_softmax(model_out[:, ix], dim=1)
        return log_pred

    def q_posterior(self, log_x_start: Tensor, log_x_t: Tensor, t: Tensor) -> Tensor:
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.to(log_x_start.device).view(-1, *num_axes) * torch.ones_like(
            log_x_start
        )
        log_EV_qxtmin_x0 = torch.where(
            t_broadcast == 0, log_x_start, log_EV_qxtmin_x0.to(torch.float32)
        )

        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = unnormed_logprobs - sliced_logsumexp(
            unnormed_logprobs, self.offsets
        )

        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, model_out: Tensor, log_x: Tensor, t: Tensor) -> Tensor:
        if self.parametrization == "x0":
            log_x_recon = self.predict_start(model_out, log_x)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t
            )
        elif self.parametrization == "direct":
            log_model_pred = self.predict_start(model_out, log_x)
        else:
            raise ValueError(f"unknown parametrization {self.parametrization}")
        return log_model_pred

    @torch.no_grad()
    def p_sample(self, model_out: Tensor, log_x: Tensor, t: Tensor) -> Tensor:
        model_log_prob = self.p_pred(model_out, log_x=log_x, t=t)
        out = self.log_sample_categorical(model_log_prob)
        return out

    def log_sample_categorical(self, logits: Tensor) -> Tensor:
        full_sample = []
        for i in range(len(self.num_classes)):
            one_class_logits = logits[:, self.slices_for_classes[i]]
            uniform = torch.rand_like(one_class_logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            sample = (gumbel_noise + one_class_logits).argmax(dim=1)
            full_sample.append(sample.unsqueeze(1))
        full_sample = torch.cat(full_sample, dim=1)
        log_sample = index_to_log_onehot(full_sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start: Tensor, t: Tensor) -> Tensor:
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def kl_prior(self, log_x_start: Tensor) -> Tensor:
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(
            self.num_classes_expanded * torch.ones_like(log_qxT_prob)
        )

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def compute_Lt(
        self,
        model_out: Tensor,
        log_x_start: Tensor,
        log_x_t: Tensor,
        t: Tensor,
        detach_mean: bool = False,
    ) -> Tensor:
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_x_t, t=t)
        log_model_prob = self.p_pred(model_out, log_x=log_x_t, t=t)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1.0 - mask) * kl

        return loss

    def sample_time(
        self, b: int, device: torch.device, method: str = "uniform"
    ) -> tuple:
        if method == "importance":
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method="uniform")

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = (Lt_sqrt / Lt_sqrt.sum()).to(device)

            t = torch.multinomial(pt_all, num_samples=b, replacement=True).to(device)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == "uniform":
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt

        else:
            raise ValueError(
                "Unknown sampling method. Must be 'importance' or 'uniform'."
            )

    def _multinomial_loss(
        self,
        model_out: Tensor,
        log_x_start: Tensor,
        log_x_t: Tensor,
        t: Tensor,
        pt: Tensor,
    ) -> Tensor:
        if self.multinomial_loss_type == "vb_stochastic":
            kl = self.compute_Lt(model_out, log_x_start, log_x_t, t)
            kl_prior = self.kl_prior(log_x_start)
            # Upweigh loss term of the kl
            vb_loss = kl / pt + kl_prior

            return vb_loss

        elif self.multinomial_loss_type == "vb_all":
            # Expensive, dont do it ;).
            # DEPRECATED
            return -self.nll(log_x_start)

        else:
            raise ValueError(
                "Unknown multinomial loss type. Must be 'vb_stochastic' or 'vb_all'."
            )

    def mixed_loss(self, x: Tensor, cond: Optional[Tensor] = None) -> tuple:
        b = x.shape[0]
        device = x.device
        t, pt = self.sample_time(b, device, "uniform")

        x_num = x[:, : self.num_numerics]
        x_cat = x[:, self.num_numerics :]

        x_num_t = x_num
        log_x_cat_t = x_cat
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = self.gaussian_q_sample(x_num, t, noise=noise)
        if x_cat.shape[1] > 0:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes)
            log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t)

        x_in = torch.cat([x_num_t, log_x_cat_t], dim=1)

        model_out = self.denoise_fn(x_in, t, y=cond)

        model_out_num = model_out[:, : self.num_numerics]
        model_out_cat = model_out[:, self.num_numerics :]

        loss_multi = torch.zeros((1,)).float()
        loss_gauss = torch.zeros((1,)).float()

        if x_cat.shape[1] > 0:
            loss_multi = self._multinomial_loss(
                model_out_cat, log_x_cat, log_x_cat_t, t, pt
            ) / len(self.num_classes)

        if x_num.shape[1] > 0:
            loss_gauss = self._gaussian_loss(model_out_num, x_num, x_num_t, t, noise)

        return loss_multi.mean(), loss_gauss.mean()

    @torch.no_grad()
    def mixed_elbo(self, x0: Tensor, cond: Optional[Tensor] = None) -> dict:
        b = x0.size(0)
        device = x0.device

        x_num = x0[:, : self.num_numerics]
        x_cat = x0[:, self.num_numerics :]
        has_cat = x_cat.shape[1] > 0
        if has_cat:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes).to(device)

        gaussian_loss = []
        xstart_mse = []
        mse = []
        mu_mse = []
        out_mean = []
        true_mean = []
        multinomial_loss = []
        for t in range(self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()
            noise = torch.randn_like(x_num)

            x_num_t = self.gaussian_q_sample(x_start=x_num, t=t_array, noise=noise)
            if has_cat:
                log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t_array)
            else:
                log_x_cat_t = x_cat

            model_out = self.denoise_fn(
                torch.cat([x_num_t, log_x_cat_t], dim=1), t_array, y=cond
            )

            model_out_num = model_out[:, : self.num_numerics]
            model_out_cat = model_out[:, self.num_numerics :]

            kl = torch.tensor([0.0])
            if has_cat:
                kl = self.compute_Lt(
                    model_out=model_out_cat,
                    log_x_start=log_x_cat,
                    log_x_t=log_x_cat_t,
                    t=t_array,
                )

            out = self._vb_terms_bpd(
                model_out_num,
                x_start=x_num,
                x_t=x_num_t,
                t=t_array,
            )

            multinomial_loss.append(kl)
            gaussian_loss.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_num) ** 2))
            mu_mse.append(mean_flat(out["mean_mse"]))
            out_mean.append(mean_flat(out["out_mean"]))
            true_mean.append(mean_flat(out["true_mean"]))

            eps = self._predict_eps_from_xstart(x_num_t, t_array, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        gaussian_loss = torch.stack(gaussian_loss, dim=1)
        multinomial_loss = torch.stack(multinomial_loss, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)
        mu_mse = torch.stack(mu_mse, dim=1)
        out_mean = torch.stack(out_mean, dim=1)
        true_mean = torch.stack(true_mean, dim=1)

        prior_gauss = self._prior_gaussian(x_num)

        prior_multin = torch.tensor([0.0])
        if has_cat:
            prior_multin = self.kl_prior(log_x_cat)

        total_gauss = torch.sum(gaussian_loss, dim=1) + prior_gauss
        total_multin = torch.sum(multinomial_loss, dim=1) + prior_multin
        return {
            "total_gaussian": total_gauss,
            "total_multinomial": total_multin,
            "losses_gaussian": gaussian_loss,
            "losses_multinimial": multinomial_loss,
            "xstart_mse": xstart_mse,
            "mse": mse,
            "mu_mse": mu_mse,
            "out_mean": out_mean,
            "true_mean": true_mean,
        }

    @torch.no_grad()
    def gaussian_ddim_step(
        self,
        model_out_num: Tensor,
        x: Tensor,
        t: Tensor,
        eta: float = 0.0,
    ) -> Tensor:
        out = self.gaussian_p_mean_variance(
            model_out_num,
            x,
            t,
            model_kwargs=None,
        )

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = perm_and_expand(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = perm_and_expand(self.alphas_cumprod_prev, t, x.shape)
        sigma = eta or (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise

        return sample

    @torch.no_grad()
    def gaussian_ddim_reverse_step(
        self,
        model_out_num: Tensor,
        x: Tensor,
        t: Tensor,
    ) -> Tensor:
        out = self.gaussian_p_mean_variance(model_out_num, x, t)

        eps = (
            perm_and_expand(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / perm_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = perm_and_expand(self.alphas_cumprod_next, t, x.shape)

        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return mean_pred

    @torch.no_grad()
    def multinomial_ddim_step(
        self, model_out_cat: Tensor, log_x_t: Tensor, t: Tensor, eta: float = 0.0
    ) -> Tensor:
        log_x0 = self.predict_start(model_out_cat, log_x_t=log_x_t)

        alpha_bar = perm_and_expand(self.alphas_cumprod, t, log_x_t.shape)
        alpha_bar_prev = perm_and_expand(self.alphas_cumprod_prev, t, log_x_t.shape)
        sigma = eta or (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        coef1 = sigma
        coef2 = alpha_bar_prev - sigma * alpha_bar
        coef3 = 1 - coef1 - coef2

        log_ps = torch.stack(
            [
                torch.log(coef1) + log_x_t,
                torch.log(coef2) + log_x0,
                torch.log(coef3) - torch.log(self.num_classes_expanded),
            ],
            dim=2,
        )

        log_prob = torch.logsumexp(log_ps, dim=2)

        out = self.log_sample_categorical(log_prob)

        return out

    @torch.no_grad()
    def sample_ddim(self, num_samples: int, cond: Any = None) -> Tensor:
        b = num_samples
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerics), device=device)

        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros(
                (b, len(self.num_classes_expanded)), device=device
            )
            log_z = self.log_sample_categorical(uniform_logits)

        for i in reversed(range(0, self.num_timesteps)):
            debug(f"Sample timestep {i:4d}", end="\r")
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self.denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(), t, y=cond
            )
            model_out_num = model_out[:, : self.num_numerics]
            model_out_cat = model_out[:, self.num_numerics :]
            z_norm = self.gaussian_ddim_step(model_out_num, z_norm, t)
            if has_cat:
                log_z = self.multinomial_ddim_step(model_out_cat, log_z, t)

        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample

    @torch.no_grad()
    def sample(self, num_samples: int, cond: Any = None) -> Tensor:
        b = num_samples
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerics), device=device)

        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros(
                (b, len(self.num_classes_expanded)), device=device
            )
            log_z = self.log_sample_categorical(uniform_logits)

        for i in reversed(range(0, self.num_timesteps)):
            debug(f"Sample timestep {i:4d}", end="\r")
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self.denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(), t, y=cond
            )
            model_out_num = model_out[:, : self.num_numerics]
            model_out_cat = model_out[:, self.num_numerics :]
            z_norm = self.gaussian_p_sample(model_out_num, z_norm, t)["sample"]
            if has_cat:
                log_z = self.p_sample(model_out_cat, log_z, t=t)

        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample

    def sample_all(
        self,
        num_samples: int,
        cond: Any = None,
        max_batch_size: int = 2000,
        ddim: bool = False,
    ) -> Tensor:
        if ddim:
            info("Sample using DDIM.")
            sample_fn = self.sample_ddim
        else:
            sample_fn = self.sample

        indices = [*range(0, num_samples, max_batch_size), num_samples]
        all_samples = []

        for i, b in enumerate(np.diff(indices)):
            c = None if cond is None else cond[indices[i] : indices[i + 1]]
            sample = sample_fn(b, c)
            if torch.any(sample.isnan()).item():
                raise ValueError("found NaNs in sample")
            all_samples.append(sample)

        return torch.cat(all_samples, dim=0)
