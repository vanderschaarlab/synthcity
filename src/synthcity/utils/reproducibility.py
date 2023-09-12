# stdlib
import random

# third party
import numpy as np
import torch


def enable_reproducible_results(random_state: int = 0) -> None:
    np.random.seed(random_state)
    try:
        torch.manual_seed(random_state)
    except BaseException:
        pass
    random.seed(random_state)
    # TODO: Implement dgl seeding, like below:
    # dgl.seed(random_state)


def clear_cache() -> None:
    try:
        torch.cuda.empty_cache()
    except BaseException:
        pass
