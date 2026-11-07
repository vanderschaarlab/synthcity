# stdlib
from typing import Any, List

# third party
import pandas as pd
import pytest
import torch
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.plugins import Plugins
from synthcity.plugins.core.models.gan import GAN
from synthcity.plugins.core.models.vae import VAE

# Plugins whose generator is built through TabularGAN or TabularVAE, with
# settings that keep training short.
PLUGINS = {
    "adsgan": {"n_iter": 2},
    "ctgan": {"n_iter": 2},
    "decaf": {"n_iter": 2, "n_iter_baseline": 2},
    "dpgan": {"n_iter": 2},
    "pategan": {"n_iter": 1, "generator_n_iter": 2, "n_teachers": 2},
    "rtvae": {"n_iter": 2},
    "tvae": {"n_iter": 2},
}


def _training_seeds(
    monkeypatch: pytest.MonkeyPatch, name: str, random_state: int
) -> List[int]:
    """Fit the plugin and return the torch seed in effect when GAN/VAE training starts."""
    seeds: List[int] = []
    for cls in (GAN, VAE):
        original = cls.fit

        def fit(self: Any, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            seeds.append(torch.initial_seed())
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "fit", fit)

    X, y = load_iris(as_frame=True, return_X_y=True)
    X["target"] = y
    plugin = Plugins().get(name, random_state=random_state, **PLUGINS[name])
    plugin.fit(pd.DataFrame(X))
    return seeds


@pytest.mark.parametrize("name", list(PLUGINS))
def test_random_state_reaches_training(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    seeds = _training_seeds(monkeypatch, name, random_state=3)

    assert seeds, f"{name} did not train a GAN or VAE"
    assert set(seeds) == {3}
