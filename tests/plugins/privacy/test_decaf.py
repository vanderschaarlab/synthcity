# stdlib
import sys
from typing import Any

# third party
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from fhelpers import generate_fixtures
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.plugins import Plugin
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins.privacy.plugin_decaf import plugin

plugin_name = "decaf"
plugin_args = {"n_iter": 50}


def gen_data_nonlinear(
    G: Any,
    base_mean: float = 0,
    base_var: float = 0.3,
    mean: float = 0,
    var: float = 1,
    SIZE: int = 10000,
    err_type: str = "normal",
    perturb: list = [],
    sigmoid: bool = True,
    expon: float = 1.1,
) -> pd.DataFrame:
    list_edges = G.edges()
    list_vertex = G.nodes()

    order = []
    for ts in nx.algorithms.dag.topological_sort(G):
        order.append(ts)

    g = []
    for v in list_vertex:
        if v in perturb:
            g.append(np.random.normal(mean, var, SIZE))
            print("perturbing ", v, "with mean var = ", mean, var)
        else:
            if err_type == "gumbel":
                g.append(np.random.gumbel(base_mean, base_var, SIZE))
            else:
                g.append(np.random.normal(base_mean, base_var, SIZE))

    for o in order:
        for edge in list_edges:
            if o == edge[1]:  # if there is an edge into this node
                if sigmoid:
                    g[edge[1]] += 1 / 1 + np.exp(-g[edge[0]])
                else:
                    g[edge[1]] += g[edge[0]] ** 2
    g = np.swapaxes(g, 0, 1)

    return pd.DataFrame(g, columns=list(map(str, list_vertex)))


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_sanity(test_plugin: Plugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_name(test_plugin: Plugin) -> None:
    assert test_plugin.name() == plugin_name


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_type(test_plugin: Plugin) -> None:
    assert test_plugin.type() == "privacy"


@pytest.mark.parametrize("test_plugin", generate_fixtures(plugin_name, plugin))
def test_plugin_hyperparams(test_plugin: Plugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 2


@pytest.mark.parametrize(
    "struct_learning_search_method",
    ["hillclimb", "d-struct"],
)
@pytest.mark.parametrize("struct_learning_score", ["k2", "bdeu"])
@pytest.mark.slow
def test_plugin_fit(
    struct_learning_search_method: str, struct_learning_score: str
) -> None:
    test_plugin = plugin(
        n_iter=50,
        struct_learning_search_method=struct_learning_search_method,
        struct_learning_score=struct_learning_score,
        struct_learning_enabled=True,
    )
    X = pd.DataFrame(load_iris()["data"])
    test_plugin.fit(GenericDataLoader(X))


@pytest.mark.parametrize(
    "test_plugin", generate_fixtures(plugin_name, plugin, plugin_args)
)
@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
def test_plugin_generate(test_plugin: Plugin) -> None:
    X = pd.DataFrame(load_iris()["data"])
    test_plugin.fit(GenericDataLoader(X))

    X_gen = test_plugin.generate()
    assert len(X_gen) == len(X)
    assert test_plugin.schema_includes(X_gen)

    X_gen = test_plugin.generate(50)
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)
    assert list(X_gen.columns) == list(X.columns)


@pytest.mark.parametrize(
    "struct_learning_search_method",
    ["hillclimb", "d-struct"],
)
def test_get_dag(struct_learning_search_method: str) -> None:
    test_plugin = plugin(
        struct_learning_enabled=True,
        struct_learning_search_method=struct_learning_search_method,
        n_iter=50,
    )

    X = pd.DataFrame(load_iris()["data"])
    dag = test_plugin.get_dag(X)

    assert len(dag) > 0


@pytest.mark.parametrize(
    "struct_learning_search_method",
    ["hillclimb", "d-struct"],
)
@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
@pytest.mark.slow
def test_plugin_generate_and_learn_dag(struct_learning_search_method: str) -> None:
    test_plugin = plugin(
        struct_learning_enabled=True,
        struct_learning_search_method=struct_learning_search_method,
        n_iter=50,
    )

    X = pd.DataFrame(load_iris()["data"])
    test_plugin.fit(GenericDataLoader(X))

    X_gen = test_plugin.generate()
    assert len(X_gen) == len(X)
    assert test_plugin.schema_includes(X_gen)

    X_gen = test_plugin.generate(50)
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)
    assert list(X_gen.columns) == list(X.columns)


@pytest.mark.parametrize("use_dag_seed", [True])
@pytest.mark.slow
def test_debiasing(use_dag_seed: bool) -> None:
    # causal structure is in dag_seed
    synthetic_dag_seed = [
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 5],
        [2, 0],
        [3, 0],
        [3, 6],
        [3, 7],
        [6, 9],
        [0, 8],
        [0, 9],
    ]
    # edge removal dictionary
    bias_dict = {"4": ["1"]}  # This removes the edge into 4 from 1.

    # DATA SETUP according to dag_seed
    G = nx.DiGraph(synthetic_dag_seed)
    data = gen_data_nonlinear(G, SIZE=1000)
    data.columns = data.columns.astype(str)

    # model initialisation and train
    test_plugin = plugin(
        struct_learning_enabled=(not use_dag_seed),
        n_iter=100,
        n_iter_baseline=200,
    )

    # DAG check before
    disc_dag_before = test_plugin.get_dag(data)
    print("Discovered DAG on real data", disc_dag_before)
    assert ("1", "4") in disc_dag_before  # the biased edge is in the DAG

    # DECAF expectes str columns/features
    train_dag_seed = []
    if use_dag_seed:
        for edge in synthetic_dag_seed:
            train_dag_seed.append([str(edge[0]), str(edge[1])])

    # Train
    test_plugin.fit(data, dag=train_dag_seed)

    # Generate
    count = 1000
    synth_data = test_plugin.generate(count, biased_edges=bias_dict)

    # DAG for synthetic data
    disc_dag_after = test_plugin.get_dag(synth_data.dataframe())
    print("Discovered DAG on synth data", disc_dag_after)
    assert ("1", "4") not in disc_dag_after  # the biased edge should be removed
