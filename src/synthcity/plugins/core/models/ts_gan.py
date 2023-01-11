# stdlib
from typing import Any, Callable, List, Optional, Tuple

# third party
import numpy as np
import torch
from pydantic import validate_arguments
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, sampler
from tqdm import tqdm

# synthcity absolute
import synthcity.logger as log
from synthcity.utils.constants import DEVICE
from synthcity.utils.reproducibility import enable_reproducible_results

# synthcity relative
from .mlp import MLP
from .ts_model import TimeSeriesModel


class TimeSeriesGAN(nn.Module):
    """
    .. inheritance-diagram:: synthcity.plugins.core.models.ts_gan.TimeSeriesGAN
        :parts: 1


    Basic TimeSeriesGAN implementation.

    Args:
        n_static_units: int,
            Number of units for the static features
        n_static_units_latent: int,
            Number of latent units for the static features
        n_temporal_units: int,
            Number of units for the temporal features
        n_temporal_window: int,
            Number of temporal observations for each subject
        n_temporal_units_latent: int,
            Number of temporal latent units
        n_units_conditional: int = 0,
            Number of conditional units
        generator_n_layers_hidden: int. Default: 1
            Number of hidden layers in the generator
        generator_n_units_hidden: int. Default: 250
            Number of hidden units in each layer of the Generator
        generator_nonlin: string, default 'elu'
            Nonlinearity to use in the generator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        generator_n_iter: int. Default: 1000
            Maximum number of iterations in the Generator.
        generator_batch_norm: bool. Default: False
            Enable/disable batch norm for the generator
        generator_dropout: float. Default: 0
            Generator Dropout value. If 0, the dropout is not used.
        generator_residual: bool. Default: True
            Use residuals for the generator
        generator_lr: float. Default: 2e-4
            Generator learning rate.
        generator_weight_decay: float. Default = 1e-3
            l2 (ridge) penalty for the generator weights.
        discriminator_n_layers_hidden: int. Default = 1
            Number of hidden layers in the discriminator
        discriminator_n_units_hidden: int. Default = 300
            Number of hidden units in each layer of the discriminator
        discriminator_nonlin: string, default = 'relu'
            Nonlinearity to use in the discriminator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        discriminator_n_iter: int
            Maximum number of iterations in the discriminator.
        discriminator_batch_norm: bool. Default: False
            Enable/disable batch norm for the discriminator
        discriminator_dropout: float. Default = 0.1
            Dropout value for the discriminator. If 0, the dropout is not used.
        discriminator_lr: float. Default = 2e-4
            learning rate for discriminator optimizer. step_size equivalent in the JAX version.
        discriminator_weight_decay: float. Default = 1e-3
            l2 (ridge) penalty for the discriminator weights.
        batch_size: int. Default = 64
            Batch size
        n_iter_print: int
            Number of iterations after which to print updates and check the validation loss.
        random_state: int
            random_state used
        clipping_value: int, default = 0
            Gradients clipping value. Zero disables the feature
        gamma_penalty: float. Default = 1.
            Latent representation penalty
        moments_penalty: float. Default = 100.
            Generator Moments(var and mean) penalty
        embedding_penalty: float. Default = 10
            Embedding representation penalty
        dataloader_sampler: Optional[sampler.Sampler] = None
            Optional data sampler
        mode: str = "RNN"
            Core neural net architecture.
            Available models:
                - "LSTM"
                - "GRU"
                - "RNN"
                - "Transformer"
                - "MLSTM_FCN"
                - "TCN"
                - "InceptionTime"
                - "InceptionTimePlus"
                - "XceptionTime"
                - "ResCNN"
                - "OmniScaleCNN"
                - "XCM"
        device
            The device used by PyTorch. cpu/cuda
        use_horizon_condition: bool. Default = True
            Whether to condition the covariate generation on the observation times or not.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_static_units: int,
        n_static_units_latent: int,
        n_temporal_units: int,
        n_temporal_window: int,
        n_temporal_units_latent: int,
        n_units_conditional: int = 0,
        generator_n_layers_hidden: int = 1,
        generator_n_units_hidden: int = 250,
        generator_nonlin: str = "leaky_relu",
        generator_static_nonlin_out: Optional[List[Tuple[str, int]]] = None,
        generator_temporal_nonlin_out: Optional[List[Tuple[str, int]]] = None,
        generator_n_iter: int = 1000,
        generator_batch_norm: bool = False,
        generator_dropout: float = 0,
        generator_loss: Optional[Callable] = None,
        generator_lr: float = 2e-4,
        generator_weight_decay: float = 1e-3,
        generator_residual: bool = True,
        discriminator_n_layers_hidden: int = 1,
        discriminator_n_units_hidden: int = 300,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_batch_norm: bool = False,
        discriminator_dropout: float = 0.1,
        discriminator_loss: Optional[Callable] = None,
        discriminator_lr: float = 2e-4,
        discriminator_weight_decay: float = 1e-3,
        batch_size: int = 64,
        n_iter_print: int = 10,
        random_state: int = 0,
        clipping_value: int = 1,
        gamma_penalty: float = 1,
        moments_penalty: float = 100,
        embedding_penalty: float = 10,
        dataloader_sampler: Optional[sampler.Sampler] = None,
        mode: str = "RNN",
        device: Any = DEVICE,
        use_horizon_condition: bool = True,
    ) -> None:
        super(TimeSeriesGAN, self).__init__()

        enable_reproducible_results(random_state)

        log.debug(f"Training GAN on device {device}")
        self.device = device

        self.n_static_units = n_static_units
        self.n_static_units_latent = n_static_units_latent
        self.n_temporal_units = n_temporal_units
        self.n_temporal_units_latent = n_temporal_units_latent
        self.n_temporal_window = n_temporal_window
        self.n_units_conditional = n_units_conditional

        # Static generator: Z_static -> static_data
        self.static_generator = MLP(
            task_type="regression",
            n_units_in=n_static_units_latent + n_units_conditional,
            n_units_out=n_static_units,
            n_layers_hidden=generator_n_layers_hidden,
            n_units_hidden=generator_n_units_hidden,
            nonlin=generator_nonlin,
            nonlin_out=generator_static_nonlin_out,
            n_iter=generator_n_iter,
            batch_norm=generator_batch_norm,
            dropout=generator_dropout,
            loss=generator_loss,
            random_state=random_state,
            lr=generator_lr,
            residual=generator_residual,
            device=device,
        )

        # Temporal horizons generator: Z_static + Z_temporal + conditional -> observation_times
        self.observation_times_generator = MLP(
            task_type="regression",
            n_units_in=n_static_units_latent
            + n_units_conditional
            + n_temporal_units_latent * n_temporal_window,
            n_units_out=n_temporal_window,
            n_layers_hidden=generator_n_layers_hidden,
            n_units_hidden=generator_n_units_hidden,
            nonlin=generator_nonlin,
            nonlin_out=[("tanh", n_temporal_window)],
            n_iter=generator_n_iter,
            batch_norm=generator_batch_norm,
            dropout=generator_dropout,
            loss=generator_loss,
            random_state=random_state,
            lr=generator_lr,
            residual=generator_residual,
            device=device,
        )

        rnn_generator_extra_args = {
            "n_static_layers_hidden": generator_n_layers_hidden,
            "n_static_units_hidden": generator_n_units_hidden,
            "n_temporal_layers_hidden": generator_n_layers_hidden,
            "n_temporal_units_hidden": generator_n_units_hidden,
            "mode": mode,
            "nonlin": generator_nonlin,
            "n_iter": generator_n_iter,
            "dropout": generator_dropout,
            "loss": generator_loss,
            "random_state": random_state,
            "lr": generator_lr,
            "device": device,
            "use_horizon_condition": use_horizon_condition,
        }
        # Embedding network between original feature space to latent space: (X_static, recovered_temporal_data) -> temporal_embeddings
        self.temporal_embedder = TimeSeriesModel(
            task_type="regression",
            n_static_units_in=n_static_units + n_units_conditional,
            n_temporal_units_in=n_temporal_units,
            n_temporal_window=n_temporal_window,
            output_shape=[n_temporal_window, n_temporal_units_latent],
            **rnn_generator_extra_args,
        ).to(self.device)

        # Recovery network from latent space to original space: (X_static, temporal_embeddings) -> recovered_temporal_data
        self.temporal_recovery = TimeSeriesModel(
            task_type="regression",
            n_static_units_in=n_static_units + n_units_conditional,
            n_temporal_units_in=n_temporal_units_latent,
            n_temporal_window=n_temporal_window,
            output_shape=[n_temporal_window, n_temporal_units],
            nonlin_out=generator_temporal_nonlin_out,
            **rnn_generator_extra_args,
        ).to(self.device)

        # Temporal generator from the latent space: Z_temporal -> E_temporal
        self.temporal_generator = TimeSeriesModel(
            task_type="regression",
            n_static_units_in=n_static_units + n_units_conditional,
            n_temporal_units_in=n_temporal_units_latent,
            n_temporal_window=n_temporal_window,
            output_shape=[n_temporal_window, n_temporal_units_latent],
            **rnn_generator_extra_args,
        ).to(self.device)

        # Temporal supervisor: Generate the next sequence: E_temporal -> fake_next_temporal_embeddings_temporal
        self.temporal_supervisor = TimeSeriesModel(
            task_type="regression",
            n_static_units_in=n_static_units + n_units_conditional,
            n_temporal_units_in=n_temporal_units_latent,
            n_temporal_window=n_temporal_window,
            output_shape=[n_temporal_window, n_temporal_units_latent],
            **rnn_generator_extra_args,
        ).to(self.device)

        # Discriminate the original and synthetic time-series data.
        self.discriminator = TimeSeriesModel(
            task_type="regression",
            n_static_units_in=n_static_units + n_units_conditional,
            n_temporal_units_in=n_temporal_units,
            n_temporal_window=n_temporal_window,
            output_shape=[1],
            n_static_layers_hidden=discriminator_n_layers_hidden,
            n_static_units_hidden=discriminator_n_units_hidden,
            n_temporal_layers_hidden=discriminator_n_layers_hidden,
            n_temporal_units_hidden=discriminator_n_units_hidden,
            nonlin=discriminator_nonlin,
            mode=mode,
            nonlin_out=[("sigmoid", 1)],
            n_iter=discriminator_n_iter,
            dropout=discriminator_dropout,
            loss=discriminator_loss,
            random_state=random_state,
            lr=discriminator_lr,
            device=device,
            use_horizon_condition=use_horizon_condition,
        ).to(self.device)

        self.discriminator_horizons = MLP(
            task_type="regression",
            n_units_in=n_temporal_window,
            n_units_out=1,
            n_layers_hidden=discriminator_n_layers_hidden,
            n_units_hidden=discriminator_n_units_hidden,
            nonlin=discriminator_nonlin,
            nonlin_out=[("sigmoid", 1)],
            n_iter=discriminator_n_iter,
            dropout=discriminator_dropout,
            loss=discriminator_loss,
            random_state=random_state,
            lr=discriminator_lr,
            device=device,
        ).to(self.device)

        # training
        self.generator_n_iter = generator_n_iter
        self.discriminator_n_iter = discriminator_n_iter
        self.n_iter_print = n_iter_print
        self.batch_size = batch_size
        self.clipping_value = clipping_value
        self.criterion = nn.BCELoss()

        self.gamma_penalty = gamma_penalty
        self.moments_penalty = moments_penalty
        self.embedding_penalty = embedding_penalty

        self.random_state = random_state

        self.dataloader_sampler = dataloader_sampler

    def fit(
        self,
        static_data: np.ndarray,
        temporal_data: np.ndarray,
        observation_times: np.ndarray,
        cond: Optional[np.ndarray] = None,
    ) -> "TimeSeriesGAN":
        static_data_t = self._check_tensor(static_data).float()
        temporal_data_t = self._check_tensor(temporal_data).float()
        observation_times_t = self._check_tensor(observation_times).float()

        condt: Optional[torch.Tensor] = None

        if self.n_units_conditional > 0:
            if cond is None:
                raise ValueError("Expecting valid conditional for training")
            if len(cond.shape) == 1:
                cond = cond.reshape(-1, 1)
            if cond.shape[1] != self.n_units_conditional:
                raise ValueError(
                    f"Expecting conditional with n_units = {self.n_units_conditional}. Got {cond.shape}"
                )
            if cond.shape[0] != static_data.shape[0]:
                raise ValueError(
                    "Expecting conditional with the same length as the dataset"
                )

            condt = self._check_tensor(cond).float()

        self._train(
            static_data_t,
            temporal_data_t,
            observation_times_t,
            condt,
        )

        return self

    def generate(
        self,
        count: int,
        cond: Optional[np.ndarray] = None,
        static_data: Optional[np.ndarray] = None,
        observation_times: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        condt: Optional[torch.Tensor] = None
        static_t: Optional[torch.Tensor] = None
        horizons_t: Optional[torch.Tensor] = None

        self.static_generator.eval()
        self.observation_times_generator.eval()
        self.temporal_generator.eval()
        self.temporal_embedder.eval()
        self.temporal_supervisor.eval()
        self.temporal_recovery.eval()

        if cond is not None:
            condt = self._check_tensor(cond).float()
        if static_data is not None:
            static_t = self._check_tensor(static_data).float()
        if observation_times is not None:
            horizons_t = self._check_tensor(observation_times).float()

        static, temporal, observation_times = self(
            count, condt, static_data=static_t, observation_times=horizons_t
        )
        return (
            static.detach().cpu().numpy(),
            temporal.detach().cpu().numpy(),
            observation_times.detach().cpu().numpy(),
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(
        self,
        count: int,
        cond: Optional[torch.Tensor] = None,
        static_data: Optional[torch.Tensor] = None,
        observation_times: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if cond is None and self.n_units_conditional > 0:
            # sample from the original conditional
            if self._original_cond is None:
                raise ValueError("Invalid original conditional. Provide a valid value.")
            cond_idxs = torch.randint(len(self._original_cond), (count,))
            cond = self._original_cond[cond_idxs]

        if cond is not None and len(cond.shape) == 1:
            cond = cond.reshape(-1, 1)

        if cond is not None and len(cond) != count:
            raise ValueError(
                f"cond length must match count Actual {len(cond)} Expected{count}"
            )

        # Static data
        if static_data is None:
            static_noise = torch.randn(
                count, self.n_static_units_latent, device=self.device
            )
            static_noise = self._append_optional_cond(static_noise, cond)
            static_data = self.static_generator(static_noise)
        static_data_with_cond = self._append_optional_cond(static_data, cond)

        # Temporal data
        temporal_noise = torch.randn(
            count,
            self.n_temporal_window,
            self.n_temporal_units_latent,
            device=self.device,
        )
        if observation_times is None:
            static_data_with_cond_and_temporal_noise = self._append_optional_cond(
                static_data_with_cond,
                temporal_noise.view(len(static_data_with_cond), -1),
            )

            observation_times = self.observation_times_generator(
                static_data_with_cond_and_temporal_noise
            )

        temporal_latent_data = self.temporal_generator(
            static_data_with_cond, temporal_noise, observation_times
        )
        fake_next_temporal_embeddings = self.temporal_supervisor(
            static_data_with_cond, temporal_latent_data, observation_times
        )
        temporal_data = self.temporal_recovery(
            static_data_with_cond, fake_next_temporal_embeddings, observation_times
        )

        return static_data, temporal_data, observation_times

    def dataloader(
        self,
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        observation_times: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> DataLoader:
        if cond is None:
            dataset = TensorDataset(static_data, temporal_data, observation_times)
        else:
            dataset = TensorDataset(static_data, temporal_data, observation_times, cond)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=self.dataloader_sampler,
            pin_memory=False,
        )

    def _train_epoch_all_models(
        self,
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        observation_times: torch.Tensor,
        cond: Optional[torch.Tensor],
    ) -> Tuple:
        batch_size = min(self.batch_size, len(static_data))

        # Prepare the real batch
        real_static_data = static_data.to(self.device)
        real_static_data = self._append_optional_cond(real_static_data, cond)

        real_observation_times = observation_times.to(self.device)

        # Prepare the fake batch
        static_noise = torch.randn(
            batch_size, self.n_static_units_latent, device=self.device
        )
        static_noise = self._append_optional_cond(static_noise, cond)

        temporal_noise = torch.randn(
            batch_size,
            self.n_temporal_window,
            self.n_temporal_units_latent,
            device=self.device,
        )

        # static data
        fake_static_data = self.static_generator(static_noise)
        fake_static_data = self._append_optional_cond(fake_static_data, cond)

        fake_static_data_with_temporal_noise = self._append_optional_cond(
            fake_static_data, temporal_noise.view(len(fake_static_data), -1)
        )
        fake_observation_times = self.observation_times_generator(
            fake_static_data_with_temporal_noise
        )

        # Embedder & Recovery
        temporal_embeddings = self.temporal_embedder(
            real_static_data, temporal_data, real_observation_times
        )
        recovered_temporal_data = self.temporal_recovery(
            real_static_data, temporal_embeddings, real_observation_times
        )

        # Generator
        temporal_latent_data = self.temporal_generator(
            fake_static_data, temporal_noise, fake_observation_times
        )
        fake_next_temporal_embeddings = self.temporal_supervisor(
            fake_static_data, temporal_latent_data, fake_observation_times
        )
        next_temporal_embeddings = self.temporal_supervisor(
            fake_static_data, temporal_embeddings, fake_observation_times
        )

        # Synthetic data
        fake_temporal_data = self.temporal_recovery(
            fake_static_data,
            fake_next_temporal_embeddings,
            fake_observation_times,
        )

        # Discriminator
        outcome_fake = self.discriminator(
            fake_static_data,
            fake_next_temporal_embeddings,
            fake_observation_times,
        ).squeeze()
        outcome_real = self.discriminator(
            real_static_data, temporal_embeddings, real_observation_times
        ).squeeze()
        outcome_latent = self.discriminator(
            fake_static_data, temporal_latent_data, fake_observation_times
        ).squeeze()
        horizons_d_fake = self.discriminator_horizons(fake_observation_times).squeeze()
        horizons_d_real = self.discriminator_horizons(real_observation_times).squeeze()

        return (
            temporal_embeddings,
            recovered_temporal_data,
            temporal_latent_data,
            fake_next_temporal_embeddings,
            next_temporal_embeddings,
            fake_temporal_data,
            outcome_fake,
            outcome_real,
            outcome_latent,
            horizons_d_fake,
            horizons_d_real,
        )

    def _train_epoch_embedding(
        self,
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        observation_times: torch.Tensor,
        cond: Optional[torch.Tensor],
    ) -> float:
        # Update the G network
        train_models = [
            self.temporal_embedder,
            self.temporal_recovery,
            self.temporal_supervisor,
        ]
        for model in train_models:
            model.optimizer.zero_grad()

        if len(static_data) != len(temporal_data):
            raise ValueError("Static and temporal lengths should be the same")

        (
            temporal_embeddings,
            recovered_temporal_data,
            temporal_latent_data,
            fake_next_temporal_embeddings,
            next_temporal_embeddings,
            fake_temporal_data,
            _,
            _,
            _,
            _,
            _,
        ) = self._train_epoch_all_models(
            static_data, temporal_data, observation_times, cond
        )

        # Embedder network loss
        errG_supervised = nn.MSELoss()(
            temporal_embeddings[:, 1:, :], next_temporal_embeddings[:, :-1, :]
        )

        errE_mse = nn.MSELoss()(temporal_data, recovered_temporal_data)
        errE_rmse = self.embedding_penalty * torch.sqrt(errE_mse)
        errE = errE_rmse + 0.1 * errG_supervised

        # Calculate gradients for G
        errE.backward()

        # Update G
        for model in train_models:
            if self.clipping_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipping_value)
            model.optimizer.step()

        # Return loss
        return errE.item()

    def _train_epoch_generator(
        self,
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        observation_times: torch.Tensor,
        cond: Optional[torch.Tensor],
    ) -> float:
        # Update the G network
        train_models = [
            self.static_generator,
            self.temporal_generator,
            self.observation_times_generator,
            self.temporal_supervisor,
        ]
        for model in train_models:
            model.optimizer.zero_grad()

        if len(static_data) != len(temporal_data):
            raise ValueError("Static and temporal lengths should be the same")

        (
            temporal_embeddings,
            recovered_temporal_data,
            temporal_latent_data,
            fake_next_temporal_embeddings,
            next_temporal_embeddings,
            fake_temporal_data,
            outcome_fake,
            outcome_real,
            outcome_latent,
            horizons_d_fake,
            horizons_d_real,
        ) = self._train_epoch_all_models(
            static_data, temporal_data, observation_times, cond
        )

        fake_labels = torch.ones(len(temporal_data)).to(self.device).squeeze()

        # Generator loss
        # 1. Adversarial loss
        errG_discrimination = self.criterion(outcome_fake, fake_labels)
        errG_discrimination_latent = self.criterion(outcome_latent, fake_labels)
        errG_discrimination_horizons = self.criterion(horizons_d_fake, fake_labels)

        # 2. Supervised loss
        errG_supervised = nn.MSELoss()(
            temporal_embeddings[:, 1:, :], next_temporal_embeddings[:, :-1, :]
        )

        # 3. Two Momments

        fake_temporal_data_var, fake_temporal_data_mean = torch.var_mean(
            fake_temporal_data, 0
        )
        temporal_var, temporal_mean = torch.var_mean(temporal_data, 0)

        errG_l1_var = nn.L1Loss()(
            torch.sqrt(fake_temporal_data_var + 1e-6), torch.sqrt(temporal_var + 1e-6)
        )
        errG_l1_mean = nn.L1Loss()(fake_temporal_data_mean, temporal_mean)
        errG_l1_moments = errG_l1_var + errG_l1_mean

        # 4. Summation
        G_loss = (
            errG_discrimination
            + errG_discrimination_horizons
            + self.gamma_penalty * errG_discrimination_latent
            + 100 * torch.sqrt(errG_supervised)
            + self.moments_penalty * errG_l1_moments
        )

        # Calculate gradients for G
        G_loss.backward()

        # Update G
        for model in train_models:
            if self.clipping_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipping_value)
            model.optimizer.step()

        # Return loss
        return G_loss.item()

    def _train_epoch_discriminator(
        self,
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        observation_times: torch.Tensor,
        cond: Optional[torch.Tensor],
    ) -> float:
        # Update the D network
        errors = []

        train_models = [self.discriminator_horizons, self.discriminator]
        for model in train_models:
            model.optimizer.zero_grad()

        (
            _,
            _,
            _,
            _,
            _,
            _,
            outcome_fake,
            outcome_real,
            outcome_latent,
            horizons_d_fake,
            horizons_d_real,
        ) = self._train_epoch_all_models(
            static_data, temporal_data, observation_times, cond
        )

        real_labels = torch.ones(len(temporal_data)).to(self.device).squeeze()
        fake_labels = torch.zeros(len(temporal_data)).to(self.device).squeeze().float()

        errD_real = self.criterion(outcome_real, real_labels)
        errD_fake = self.criterion(outcome_fake, fake_labels)
        errD_fake_e = self.criterion(outcome_latent, fake_labels)
        errD_horizon_fake = self.criterion(horizons_d_fake, fake_labels)
        errD_horizon_real = self.criterion(horizons_d_real, real_labels)

        errD = (
            errD_real
            + errD_fake
            + errD_horizon_fake
            + errD_horizon_real
            + self.gamma_penalty * errD_fake_e
        )

        errD.backward()
        # Update D
        for model in train_models:
            if self.clipping_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipping_value)
            model.optimizer.step()

        errors.append(errD.item())

        return np.mean(errors)

    def _train_epoch(
        self,
        loader: DataLoader,
    ) -> Tuple[float, float, float]:
        E_losses = []
        G_losses = []
        D_losses = []

        for i, data in enumerate(loader):
            cond: Optional[torch.Tensor] = None
            if self.n_units_conditional > 0:
                static_data, temporal_data, observation_times, cond = data
            else:
                static_data, temporal_data, observation_times = data

            E_losses.append(
                self._train_epoch_embedding(
                    static_data,
                    temporal_data,
                    observation_times,
                    cond,
                )
            )
            D_losses.append(
                self._train_epoch_discriminator(
                    static_data,
                    temporal_data,
                    observation_times,
                    cond,
                )
            )
            G_losses.append(
                self._train_epoch_generator(
                    static_data,
                    temporal_data,
                    observation_times,
                    cond,
                )
            )

        return np.mean(E_losses), np.mean(G_losses), np.mean(D_losses)

    def _train(
        self,
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        observation_times: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> "TimeSeriesGAN":
        self._original_cond = cond
        static_data = self._check_tensor(static_data).float()
        temporal_data = self._check_tensor(temporal_data).float()
        observation_times = self._check_tensor(observation_times).float()

        # Load Dataset
        loader = self.dataloader(static_data, temporal_data, observation_times, cond)

        # Train loop
        for i in tqdm(range(self.generator_n_iter)):
            e_loss, g_loss, d_loss = self._train_epoch(
                loader,
            )
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i + 1) % self.n_iter_print == 0:
                log.debug(
                    f"[{i}/{self.generator_n_iter}]\tLoss_D: {d_loss}\tLoss_G: {g_loss}\t Loss_E: {e_loss}"
                )

        return self

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)

    def _append_optional_cond(
        self, X: torch.Tensor, cond: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if cond is None:
            return X

        return torch.cat([X, cond], dim=1)
