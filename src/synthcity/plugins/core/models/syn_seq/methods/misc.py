# methods/misc.py

# stdlib
from typing import Any, Dict, Optional

# third party
import numpy as np

###############################################################################
# LOGNORM
###############################################################################


def syn_lognorm(
    y: np.ndarray, X: np.ndarray, random_state: int = 0, **kwargs: Any
) -> Dict[str, Any]:
    """
    Fit a 'lognormal' model. Typically you'd compute log of y and estimate
    parameters. This is just a stub/demonstration.

    Args:
        y: 1D array of target values.
        X: 2D array of predictors (not strictly used here).
        random_state: random seed for reproducibility.
        **kwargs: any extra hyperparameters you might use in a real model.

    Returns:
        model_info: dictionary containing the fitted model parameters.
    """
    # To avoid log of zero/negative values, shift if needed
    shift = max(0, -y.min() + 1e-9)
    y_log = np.log(y + shift)

    mu = float(np.mean(y_log))
    sigma = float(np.std(y_log)) if np.std(y_log) > 0 else 1.0

    model_info = {
        "model_type": "lognorm",
        "shift": shift,
        "mu": mu,
        "sigma": sigma,
        "random_state": random_state,
    }
    return model_info


def generate_lognorm(
    fitted_model: Dict[str, Any],
    X: np.ndarray,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Generate new y from the previously fitted lognormal parameters.

    Args:
        fitted_model: The dictionary returned by syn_lognorm(...).
        X: 2D array of predictors (not used here, but included for consistency).
        random_state: to override the model's stored random seed.
        **kwargs: extra arguments for generation.

    Returns:
        y_gen: 1D numpy array of synthesized target values.
    """
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng(fitted_model["random_state"])

    shift = fitted_model["shift"]
    mu = fitted_model["mu"]
    sigma = fitted_model["sigma"]

    n = len(X)
    # Sample from lognormal
    y_gen = rng.lognormal(mean=mu, sigma=sigma, size=n)
    # Shift back
    y_gen -= shift

    return y_gen


###############################################################################
# RANDOM
###############################################################################


def syn_random(
    y: np.ndarray, X: np.ndarray, random_state: int = 0, **kwargs: Any
) -> Dict[str, Any]:
    """
    Fit a 'random' approach: just store the min and max of y,
    so we can sample uniformly in [min, max].

    Args:
        y: 1D array of target values.
        X: 2D array of predictors (not used here).
        random_state: random seed.
        **kwargs: placeholders for advanced usage.

    Returns:
        model_info: dictionary with min_val, max_val, etc.
    """
    min_val = float(np.min(y))
    max_val = float(np.max(y))

    model_info = {
        "model_type": "random",
        "min_val": min_val,
        "max_val": max_val,
        "random_state": random_state,
    }
    return model_info


def generate_random(
    fitted_model: Dict[str, Any],
    X: np.ndarray,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Generate y uniformly between [min_val, max_val].

    Args:
        fitted_model: The dictionary returned by syn_random(...).
        X: 2D array of predictors (not used here).
        random_state: optional seed.
        **kwargs: unused extras.

    Returns:
        y_syn: 1D numpy array of uniformly sampled values.
    """
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng(fitted_model["random_state"])

    min_val = fitted_model["min_val"]
    max_val = fitted_model["max_val"]

    n = len(X)
    y_syn = rng.uniform(low=min_val, high=max_val, size=n)
    return y_syn


###############################################################################
# SWR (Sampling Without Replacement)
###############################################################################


def syn_swr(
    y: np.ndarray, X: np.ndarray, random_state: int = 0, **kwargs: Any
) -> Dict[str, Any]:
    """
    Fit a 'sampling without replacement' approach. We store a random permutation
    of y. Then we can generate slices from it without replacement.

    Args:
        y: 1D array of target values.
        X: 2D array of predictors (unused here).
        random_state: random seed.
        **kwargs: additional arguments if needed.

    Returns:
        model_info: dictionary containing the permuted pool of y and a pointer
                    for how many draws have been used so far.
    """
    rng = np.random.default_rng(random_state)
    perm_indices = rng.permutation(len(y))

    model_info = {
        "model_type": "swr",
        "pool": y.copy(),
        "shuffled_idx": perm_indices,
        "current_pos": 0,
        "random_state": random_state,
    }
    return model_info


def generate_swr(
    fitted_model: Dict[str, Any],
    X: np.ndarray,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Generate new y by drawing from the 'pool' of original values WITHOUT REPLACEMENT.
    If the aggregator calls this multiple times, we continue from where we left off.

    Args:
        fitted_model: The dictionary from syn_swr(...).
        X: 2D array of predictors (not used in this simple approach).
        random_state: If you want to override the seed, but typically
                      we rely on the existing permutation.
        **kwargs: any additional arguments.

    Returns:
        y_syn: 1D array of values from the pool, sampled without replacement.

    Raises:
        ValueError: if we request more samples than remain in the pool.
    """
    # The random_state argument does not affect the existing permutation.
    # The permutation was decided at fit-time.

    pool = fitted_model["pool"]
    shuffled_idx = fitted_model["shuffled_idx"]
    current_pos = fitted_model["current_pos"]

    n = len(X)
    remaining = len(pool) - current_pos

    if n > remaining:
        raise ValueError(
            f"SWR error: requested {n} samples but only {remaining} remain. "
            "Increase your dataset or reduce requested size."
        )

    selected_idx = shuffled_idx[current_pos : current_pos + n]
    fitted_model["current_pos"] += n  # move pointer

    # Return these pool values
    y_syn = pool[selected_idx]
    return y_syn
