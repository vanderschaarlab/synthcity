from .gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion # noqa
from .modules import MLPDiffusion, ResNetDiffusion # noqa

# stdlib
from copy import deepcopy
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

# synthcity relative
from ..tabular_encoder import TabularEncoder


# class TabDDPM(nn.Module):
#     def __init__(
#         self, 
#         X: pd.DataFrame,
        
#     def generate(self, n_samples: int) -> pd.DataFrame:
#         self.eval()
#         with torch.no_grad():
#             samples = self.diffusion.sample(n_samples)
#         return samples
    
#     def forward(self, count: int) -> pd.DataFrame:
#         pass