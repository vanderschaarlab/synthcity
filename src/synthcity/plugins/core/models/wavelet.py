# stdlib
from typing import Any, List, Tuple

# third party
import numpy as np
import pywt
import torch
from torch import nn

# synthcity absolute
from synthcity.utils.constants import DEVICE

# synthcity relative
from .layers import Permute
from .mlp import MLP


class WaveBlock(nn.Module):
    def __init__(
        self, seq_len: int, wavelet: str = "db4", device: Any = DEVICE
    ) -> None:
        super(WaveBlock, self).__init__()
        w = pywt.Wavelet(wavelet)
        self.h_filter = w.dec_hi
        self.l_filter = w.dec_lo
        self.device = device

        self.mWDN_H = nn.Linear(seq_len, seq_len)
        self.mWDN_L = nn.Linear(seq_len, seq_len)
        self.mWDN_H.weight = nn.Parameter(self.create_W(seq_len, self.h_filter))
        self.mWDN_L.weight = nn.Parameter(self.create_W(seq_len, self.l_filter))
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AvgPool1d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hp_1 = self.sigmoid(self.mWDN_H(x))
        lp_1 = self.sigmoid(self.mWDN_L(x))
        hp_out = self.pool(hp_1)
        lp_out = self.pool(lp_1)
        all_out = torch.cat((hp_out, lp_out), dim=-1)
        return lp_out, all_out

    def create_W(self, size: int, filter_list: List) -> torch.Tensor:
        max_epsilon = np.min(np.abs(filter_list))
        weight_np = np.random.randn(size, size) * 0.1 * max_epsilon
        for i in range(0, size):
            filter_index = 0
            for j in range(i, size):
                if filter_index < len(filter_list):
                    weight_np[i][j] = filter_list[filter_index]
                    filter_index += 1
        return torch.from_numpy(weight_np).float().to(self.device)


class Wavelet(nn.Module):
    def __init__(
        self,
        n_units_in: int,
        n_units_window: int,
        n_units_hidden: int,
        n_layers_hidden: int = 2,
        dropout: float = 0,
        levels: int = 5,
        wavelet: str = "haar",
        device: Any = DEVICE,
    ) -> None:
        super(Wavelet, self).__init__()
        self.levels = levels
        self.device = device

        self.blocks = nn.ModuleList()
        n_unit_interm = 0
        for i in range(levels):
            width = n_units_window // 2**i
            if width < 2:
                continue
            n_unit_interm += width if width % 2 == 0 else width - 1
            self.blocks.append(WaveBlock(width, wavelet=wavelet, device=device))

        self.n_unit_interm = n_unit_interm
        self.n_units_in = n_units_in
        self.n_units_window = n_units_window
        self.n_units_hidden = n_units_hidden

        self.post_process = MLP(
            task_type="regression",
            n_units_in=n_unit_interm * n_units_in,
            n_units_out=n_units_window * n_units_hidden,
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            residual=True,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((x.shape[0], x.shape[2], 0))
        x = x.to(self.device)
        x = Permute(0, 2, 1)(x)  # bs x seq_len x nvars -> bs x nvars x seq_len
        for block in self.blocks:
            x, out_ = block(x)
            out = torch.cat((out, out_), dim=-1)
        out = Permute(0, 2, 1)(out)  # bs x outlen x seq_len -> bs x seq_len x outlen
        return self.post_process(
            out.reshape(-1, self.n_unit_interm * self.n_units_in)
        ).reshape(-1, self.n_units_window, self.n_units_hidden)
