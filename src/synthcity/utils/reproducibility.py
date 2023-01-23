# stdlib
import random

# third party
import numpy as np
import torch


def enable_reproducible_results(random_state: int = 0) -> None:
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    random.seed(random_state)


def clear_cache() -> None:
    torch.cuda.empty_cache()
