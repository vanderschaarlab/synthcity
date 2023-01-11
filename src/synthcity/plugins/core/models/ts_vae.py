# stdlib
from typing import Any, List, Optional, Tuple

# third party
import numpy as np
import torch
from pydantic import validate_arguments
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# synthcity absolute
import synthcity.logger as log
from synthcity.utils.constants import DEVICE

# synthcity relative
from .mlp import MLP
from .ts_model import TimeSeriesModel
from .vae import Decoder, Encoder


class LatentODE(nn.Module):
    def __init__(
        self,
        n_units_embedding: int,
        n_layers_hidden: int = 1,
        n_units_hidden: int = 100,
        nonlin: str = "relu",
        random_state: int = 0,
        dropout: float = 0.1,
        batch_norm: bool = True,
        residual: bool = False,
        device: Any = DEVICE,
    ) -> None:
        super(LatentODE, self).__init__()
        self.device = device
        self.model = MLP(
            task_type="regression",
            n_units_in=n_units_embedding,
            n_units_out=n_units_embedding,
            n_units_hidden=n_units_hidden,
            n_layers_hidden=n_layers_hidden,
            nonlin=nonlin,
            random_state=random_state,
            dropout=dropout,
            batch_norm=batch_norm,
            residual=residual,
            device=device,
        ).to(self.device)

    def forward(self, observation_times: torch.Tensor, temporal: Tensor) -> Tensor:
        return self.model(temporal)


class TimeSeriesEncoder(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_static_units: int,
        n_static_units_embedding: int,
        n_temporal_units: int,
        n_temporal_window: int,
        n_temporal_units_embedding: int,
        n_units_hidden: int = 200,
        n_layers_hidden: int = 2,
        nonlin: str = "relu",
        batch_norm: bool = False,
        dropout: float = 0.1,
        random_state: int = 0,
        mode: str = "LSTM",
        device: Any = DEVICE,
    ) -> None:
        super(TimeSeriesEncoder, self).__init__()
        self.mode = mode
        self.device = device

        self.static_encoder = Encoder(
            n_static_units,
            n_static_units_embedding,
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            nonlin=nonlin,
            batch_norm=batch_norm,
            dropout=dropout,
            device=device,
        )

        self.temporal_encoder = TimeSeriesModel(
            task_type="regression",
            n_static_units_in=n_static_units,
            n_temporal_units_in=n_temporal_units,
            n_temporal_window=n_temporal_window,
            output_shape=[2, n_temporal_window, n_temporal_units_embedding],
            n_static_layers_hidden=n_layers_hidden,
            n_static_units_hidden=n_units_hidden,
            n_temporal_layers_hidden=n_layers_hidden,
            n_temporal_units_hidden=n_units_hidden,
            mode=mode,
            nonlin=nonlin,
            dropout=dropout,
            random_state=random_state,
            device=device,
            use_horizon_condition=True,
        )
        self.horizons_encoder = Encoder(
            n_temporal_window,
            n_temporal_units_embedding,
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            nonlin=nonlin,
            batch_norm=batch_norm,
            dropout=dropout,
            device=device,
        )

    def forward(
        self,
        static: Tensor,
        temporal: Tensor,
        observation_times: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        static_mu, static_logvar = self.static_encoder(static)
        horizons_mu, horizons_logvar = self.horizons_encoder(observation_times)

        temporal_embs = self.temporal_encoder(static, temporal, observation_times)
        temporal_mu, temporal_logvar = (
            temporal_embs[:, 0, :].squeeze(),
            temporal_embs[:, 1, :].squeeze(),
        )

        return (
            static_mu,
            static_logvar,
            temporal_mu,
            temporal_logvar,
            horizons_mu,
            horizons_logvar,
        )


class TimeSeriesDecoder(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_static_units_embedding: int,
        n_static_units_out: int,
        n_temporal_units_embedding: int,
        n_temporal_units_out: int,
        n_temporal_window: int,
        n_layers_hidden: int = 1,
        n_units_hidden: int = 100,
        nonlin: str = "relu",
        static_nonlin_out: Optional[List[Tuple[str, int]]] = None,
        temporal_nonlin_out: Optional[List[Tuple[str, int]]] = None,
        random_state: int = 0,
        dropout: float = 0.1,
        mode: str = "LSTM",
        batch_norm: bool = False,
        residual: bool = False,
        device: Any = DEVICE,
    ) -> None:
        super(TimeSeriesDecoder, self).__init__()
        self.device = device
        self.mode = mode

        self.static_decoder = Decoder(
            n_static_units_embedding,
            n_static_units_out,
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            nonlin=nonlin,
            nonlin_out=static_nonlin_out,
            random_state=random_state,
            dropout=dropout,
            batch_norm=batch_norm,
            residual=residual,
            device=device,
        ).to(device)

        self.horizons_decoder = Decoder(
            n_temporal_units_embedding,
            n_temporal_window,
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            nonlin=nonlin,
            random_state=random_state,
            dropout=dropout,
            batch_norm=batch_norm,
            residual=residual,
            device=device,
        ).to(device)

        self.temporal_decoder = TimeSeriesModel(
            task_type="regression",
            n_static_units_in=n_static_units_out,
            n_temporal_units_in=n_temporal_units_embedding,
            n_temporal_window=n_temporal_window,
            output_shape=[n_temporal_window, n_temporal_units_out],
            n_static_layers_hidden=n_layers_hidden,
            n_static_units_hidden=n_units_hidden,
            n_temporal_layers_hidden=n_layers_hidden,
            n_temporal_units_hidden=n_units_hidden,
            mode=mode,
            nonlin=nonlin,
            nonlin_out=temporal_nonlin_out,
            dropout=dropout,
            random_state=random_state,
            device=device,
            use_horizon_condition=True,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(
        self, static_embs: Tensor, temporal_embs: Tensor, horizon_embs: Tensor
    ) -> Tensor:
        static_decoded = self.static_decoder(static_embs)
        observation_times = self.horizons_decoder(horizon_embs)

        temporal_decoded = self.temporal_decoder(
            static_decoded, temporal_embs, observation_times
        )
        return static_decoded, temporal_decoded, observation_times


class TimeSeriesVAE(nn.Module):
    """
    .. inheritance-diagram:: synthcity.plugins.core.models.ts_vae.TimeSeriesVAE
        :parts: 1

    Basic Time-Series Variational AutoEncoder

    Args:
        n_static_units: int
            Number of input static units.
        n_static_units_embedding: int.
            Number of latent static units.
        n_temporal_units: int.
            Number of input temporal units.
        n_temporal_window: int
            The length of time series sequences.
        n_temporal_units_embedding: int.
            Number of latent temporal units.
        batch_size: int. Default = 100
            Batch size.
        n_iter: int. Default = 500
            Number of epochs
        n_iter_print: int = 10.
            Frequency of printing the training loss.
        random_state: int = 0
            Random seed
        clipping_value: int = 1
            Gradients clipping value. Zero disables the feature
        lr: float = 2e-4,
            Learning rate
        weight_decay: float = 1e-3,
            l2 (ridge) penalty for the weights.
        loss_factor: int. Default = 2
            Reconstruction loss weight.
        decoder_n_layers_hidden: int. Default = 2
            Number of hidden layer in the decoder
        decoder_n_units_hidden: int. Decoder = 250.
            Number of hidden units in the decoder
        decoder_nonlin: str. Decoder = "leaky_relu"
            Activation for the hidden layers. Can be relu, elu, leaky_relu, selu.
        decoder_static_nonlin_out: Optional[List[Tuple[str, int]]] = None
            (Optional) Activations to use in the output layer of the decoder for static features.
        decoder_temporal_nonlin_out: Optional[List[Tuple[str, int]]] = None,
            (Optional) Activations to use in the output layer of the decoder for temporal features.
        decoder_batch_norm: bool. Default = False
            Decoder batch norm
        decoder_dropout: float. Default = 0
            Decoder dropout
        decoder_residual: bool. Default = True
            Use residual connections in the decoder
        decoder_mode: str. Default = "LSTM"
            Core neural net architecture for the decoder.
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
        encoder_n_layers_hidden: int. Default = 3
            Number of hidden layers in the encoder
        encoder_n_units_hidden. Default int = 300
            Number of hidden units in the encoder
        encoder_nonlin: str. Default = "leaky_relu"
            Activations for the hidden layers in the encoder.
        encoder_batch_norm: bool. Default = False,
            Encoder batch norm.
        encoder_dropout: float. Default = 0.1
            Encoder dropout.
        encoder_mode: str. Default = "LSTM"
            Core neural net architecture for the encoder.
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
                - "Transformer"
        device
            PyTorch device: cpu/cuda.
    """

    def __init__(
        self,
        n_static_units: int,
        n_static_units_embedding: int,
        n_temporal_units: int,
        n_temporal_window: int,
        n_temporal_units_embedding: int,
        batch_size: int = 100,
        n_iter: int = 500,
        n_iter_print: int = 10,
        random_state: int = 0,
        clipping_value: int = 1,
        lr: float = 2e-4,
        weight_decay: float = 1e-3,
        loss_factor: int = 2,
        # Decoder
        decoder_n_layers_hidden: int = 2,
        decoder_n_units_hidden: int = 250,
        decoder_nonlin: str = "leaky_relu",
        decoder_static_nonlin_out: Optional[List[Tuple[str, int]]] = None,
        decoder_temporal_nonlin_out: Optional[List[Tuple[str, int]]] = None,
        decoder_batch_norm: bool = False,
        decoder_dropout: float = 0,
        decoder_residual: bool = True,
        decoder_mode: str = "LSTM",
        # Encoder
        encoder_n_layers_hidden: int = 3,
        encoder_n_units_hidden: int = 300,
        encoder_nonlin: str = "leaky_relu",
        encoder_batch_norm: bool = False,
        encoder_dropout: float = 0.1,
        encoder_mode: str = "LSTM",
        device: Any = DEVICE,
    ):
        super(TimeSeriesVAE, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_iter_print = n_iter_print
        self.clipping_value = clipping_value
        self.loss_factor = loss_factor
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_static_units_embedding = int(n_static_units_embedding)
        self.n_temporal_units_embedding = int(n_temporal_units_embedding)
        self.static_nonlin_out = decoder_static_nonlin_out
        self.temporal_nonlin_out = decoder_temporal_nonlin_out
        self.n_temporal_window = n_temporal_window

        self.random_state = random_state
        torch.manual_seed(self.random_state)

        self.batch_size = batch_size
        self.n_iter = n_iter

        self.encoder = TimeSeriesEncoder(
            n_static_units=n_static_units,
            n_static_units_embedding=n_static_units_embedding,
            n_temporal_units=n_temporal_units,
            n_temporal_window=n_temporal_window,
            n_temporal_units_embedding=n_temporal_units_embedding,
            n_units_hidden=encoder_n_units_hidden,
            n_layers_hidden=encoder_n_layers_hidden,
            nonlin=encoder_nonlin,
            batch_norm=encoder_batch_norm,
            dropout=encoder_dropout,
            random_state=random_state,
            mode=encoder_mode,
            device=device,
        ).to(device)
        self.decoder = TimeSeriesDecoder(
            n_static_units_embedding=n_static_units_embedding,
            n_static_units_out=n_static_units,
            n_temporal_units_embedding=n_temporal_units_embedding,
            n_temporal_units_out=n_temporal_units,
            n_temporal_window=n_temporal_window,
            n_layers_hidden=decoder_n_layers_hidden,
            n_units_hidden=decoder_n_units_hidden,
            nonlin=decoder_nonlin,
            static_nonlin_out=decoder_static_nonlin_out,
            temporal_nonlin_out=decoder_temporal_nonlin_out,
            random_state=random_state,
            dropout=decoder_dropout,
            mode=decoder_mode,
            batch_norm=decoder_batch_norm,
            residual=decoder_residual,
            device=device,
        ).to(device)

    def fit(
        self, static: np.ndarray, temporal: np.ndarray, observation_times: np.ndarray
    ) -> Any:
        static_t = self._check_tensor(static).float()
        temporal_t = self._check_tensor(temporal).float()
        horizons_t = self._check_tensor(observation_times).float()

        self._train(static_t, temporal_t, horizons_t)

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate(self, samples: int) -> np.ndarray:
        self.decoder.eval()

        steps = samples // self.batch_size + 1

        static_data = []
        temporal_data = []
        horizons = []
        for _ in range(steps):
            static_noise = torch.randn(
                self.batch_size, self.n_static_units_embedding, device=self.device
            )
            horizons_noise = torch.randn(
                self.batch_size, self.n_temporal_units_embedding, device=self.device
            )
            temporal_noise = torch.randn(
                self.batch_size,
                self.n_temporal_window,
                self.n_temporal_units_embedding,
                device=self.device,
            )

            static_gen, temporal_gen, horizons_gen = self.decoder(
                static_noise, temporal_noise, horizons_noise
            )

            static_data.append(static_gen.detach().cpu().numpy())
            temporal_data.append(temporal_gen.detach().cpu().numpy())
            horizons.append(horizons_gen.detach().cpu().numpy())

        static_data = np.concatenate(static_data, axis=0)
        static_data = static_data[:samples]

        temporal_data = np.concatenate(temporal_data, axis=0)
        temporal_data = temporal_data[:samples]

        horizons = np.concatenate(horizons, axis=0)
        horizons = horizons[:samples]

        return static_data, temporal_data, horizons

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def _train_step(
        self, static: Tensor, temporal: Tensor, observation_times: Tensor
    ) -> Tensor:
        # Encode
        (
            static_mu,
            static_logvar,
            temporal_mu,
            temporal_logvar,
            horizons_mu,
            horizons_logvar,
        ) = self.encoder.forward(static, temporal, observation_times)
        static_embedding = self._reparameterize(static_mu, static_logvar)
        temporal_embedding = self._reparameterize(temporal_mu, temporal_logvar)
        horizons_embedding = self._reparameterize(horizons_mu, horizons_logvar)

        # Decode
        static_recon, temporal_recon, horizons_recon = self.decoder(
            static_embedding, temporal_embedding, horizons_embedding
        )

        static_loss = self._loss_function(
            static_recon, static, static_mu, static_logvar, self.static_nonlin_out
        )
        temporal_loss = self._loss_function(
            temporal_recon,
            temporal,
            temporal_mu,
            temporal_logvar,
            self.temporal_nonlin_out,
        )
        horizons_loss = self._loss_function(
            horizons_recon, observation_times, horizons_mu, horizons_logvar
        )

        loss = static_loss + temporal_loss + horizons_loss

        if torch.isnan(loss):
            raise RuntimeError("The loss contains NaNs")

        loss.backward()

        return loss.item()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _train(
        self, static: Tensor, temporal: Tensor, observation_times: Tensor
    ) -> Any:
        loader = self._dataloader(static, temporal, observation_times)

        optimizer = Adam(
            self.parameters(),
            weight_decay=self.weight_decay,
            lr=self.lr,
        )

        losses = []
        for epoch in range(self.n_iter):
            for id_, data in enumerate(loader):
                optimizer.zero_grad()

                static_mb, temporal_mb, horiz_mb = data

                losses.append(self._train_step(static_mb, temporal_mb, horiz_mb))

                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)

                optimizer.step()
            if epoch % self.n_iter_print == 0:
                log.info(f"[{epoch}/{self.n_iter}] Loss: {np.mean(losses)}")

    def _check_tensor(self, X: Tensor) -> Tensor:
        if isinstance(X, Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)

    def _dataloader(
        self, static: Tensor, temporal: Tensor, observation_times: Tensor
    ) -> DataLoader:
        dataset = TensorDataset(static, temporal, observation_times)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=False,
        )

    def _loss_function(
        self,
        reconstructed: Tensor,
        real: Tensor,
        mu: Tensor,
        logvar: Tensor,
        nonlin_out: Optional[list] = None,
    ) -> Tensor:
        if real.shape[-1] == 0:
            return 0

        if nonlin_out is not None:
            step = 0
            loss = []
            for activation, length in nonlin_out:
                step_end = step + length
                if activation == "softmax":
                    recon_slice = reconstructed[..., step:step_end]
                    if len(recon_slice.shape) == 3:
                        recon_slice = recon_slice.permute(
                            0, 2, 1
                        )  # batches, classes, seq_len
                    discr_loss = nn.NLLLoss(reduction="sum")(
                        torch.log(recon_slice + 1e-8),
                        torch.argmax(real[..., step:step_end], dim=-1),
                    )
                    loss.append(discr_loss)
                else:
                    diff = reconstructed[..., step:step_end] - real[..., step:step_end]
                    cont_loss = (50 * diff**2).sum()

                    loss.append(cont_loss)
                step = step_end

            if step != reconstructed.size()[-1]:
                raise RuntimeError(
                    f"Invalid reconstructed features. Expected {step}, got {reconstructed.shape}"
                )

            reconstruction_loss = torch.sum(torch.stack(loss))
        else:
            reconstruction_loss = nn.MSELoss()(reconstructed, real)

        KLD_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp()))

        return reconstruction_loss * self.loss_factor + KLD_loss
