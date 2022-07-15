# stdlib
from typing import Any, Dict, List, Tuple

# third party
import torch
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from torch import Tensor, nn

# synthcity absolute
from synthcity.utils.constants import DEVICE

# synthcity relative
from .layers import Permute
from .transformer import TransformerModel


class Wavelet(nn.Module):
    def __init__(
        self,
        n_units_in: int,
        n_units_window: int,
        n_units_hidden: int,
        n_layers_hidden: int = 2,
        dropout: float = 0,
        levels: int = 3,
        wavelet: str = "sym2",
        mode: str = "symmetric",
        device: Any = DEVICE,
    ) -> None:
        super(Wavelet, self).__init__()
        self.levels = levels
        self.device = device

        self.n_units_in = n_units_in
        self.n_units_window = n_units_window
        self.n_units_hidden = n_units_hidden

        self.wavelet_encoder = DWT1DForward(J=levels, wave=wavelet, mode=mode)
        self.wavelet_decoder = DWT1DInverse(wave=wavelet, mode=mode)

        self.post_processing = TransformerModel(
            n_units_in,
            n_units_hidden,
            n_layers_hidden=n_layers_hidden,
            device=device,
            dropout=dropout,
        )
        self.normalizer: Dict[int, nn.Module] = {}
        self.to(device)

    def encode(self, X: Tensor) -> Tuple[Tensor, List]:
        low, high = self.wavelet_encoder(X)

        return low, high

    def decode(self, low: Tensor, high: List[Tensor]) -> Tensor:
        decoded = self.wavelet_decoder((low, high))

        return decoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = Permute(0, 2, 1)(x)  # bs x seq_len x nvars -> bs x nvars x seq_len

        low, high = self.encode(x)

        out = torch.concat([low] + high, axis=-1).to(self.device)
        if out.shape[-1] not in self.normalizer:
            self.normalizer[out.shape[-1]] = nn.Linear(
                out.shape[-1], self.n_units_window
            )
        out = self.normalizer[out.shape[-1]](out)

        out = Permute(0, 2, 1)(out)  # bs x outlen x seq_len -> bs x seq_len x outlen
        return self.post_processing(out)
