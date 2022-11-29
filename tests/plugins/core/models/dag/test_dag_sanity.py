# third party
import igraph as ig
import numpy as np
import pandas as pd

# synthcity absolute
from synthcity.plugins.core.models.dag.dstruct import get_dstruct_dag
from synthcity.plugins.generic.plugin_bayesian_network import plugin


def simulate_dag(d: int, s0: int, graph_type: str = "ER") -> pd.DataFrame:
    """Simulate random DAG with some expected number of edges.
    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP
    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """

    def _random_permutation(M: np.ndarray) -> np.ndarray:
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und: np.ndarray) -> np.ndarray:
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G: np.ndarray) -> np.ndarray:
        return np.array(G.get_adjacency().data)

    if graph_type == "ER":
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == "SF":
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == "BP":
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError("unknown graph type")
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return pd.DataFrame(B_perm)


def test_sanity() -> None:
    ref_model = plugin()
    X = simulate_dag(5, 9)  # get_airfoil_dataset() #pd.DataFrame(load_iris()["data"])

    dag = get_dstruct_dag(X)
    ref_dag = list(ref_model._get_dag(X).edges())

    print(dag, ref_dag)
    assert len(dag) > 0
    assert len(ref_dag) > 0
