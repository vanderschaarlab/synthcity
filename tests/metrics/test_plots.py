# stdlib
from typing import Callable

# third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics.plots import plot_marginal_comparison, plot_tsne
from synthcity.plugins import Plugin, Plugins
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader


def _eval_plugin(cbk: Callable, X: DataLoader, X_syn: DataLoader) -> None:
    cbk(plt, X, X_syn)

    sz = len(X_syn)
    X_rnd = GenericDataLoader(
        pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    )

    cbk(
        plt,
        X,
        X_rnd,
    )


@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
def test_plot_marginal_comparison(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X)

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(1000)

    _eval_plugin(plot_marginal_comparison, Xloader, X_gen)


@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
def test_plot_tsne(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X)

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(1000)

    _eval_plugin(plot_tsne, Xloader, X_gen)
