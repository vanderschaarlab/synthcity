# third party
import igraph as ig
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit as sigmoid

# synthcity absolute
from synthcity.plugins.core.models.dag.dstruct import get_dstruct_dag
from synthcity.plugins.core.models.dag.utils import count_accuracy


def simulate_nonlinear_sem(B: np.ndarray, n: int, sem_type: str = "mim") -> np.ndarray:
    """Simulate samples from nonlinear SEM.
    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones
    Returns:
        X (np.ndarray): [n, d] sample matrix
    """

    def _simulate_single_equation(X: np.ndarray, scale: float) -> np.ndarray:
        """X: [n, num of parents], x: [n]"""
        z = np.random.normal(scale=scale, size=n)
        pa_size = X.shape[1]
        if pa_size == 0:
            return z
        if sem_type == "mlp":
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = sigmoid(X @ W1) @ W2 + z
        elif sem_type == "mim":
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        elif sem_type == "gp":
            # third party
            from sklearn.gaussian_process import GaussianProcessRegressor

            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == "gp-add":
            # third party
            from sklearn.gaussian_process import GaussianProcessRegressor

            gp = GaussianProcessRegressor()
            x = (
                sum(
                    [
                        gp.sample_y(X[:, i, None], random_state=None).flatten()
                        for i in range(X.shape[1])
                    ]
                )
                + z
            )
        else:
            raise ValueError("unknown sem type")
        return x

    d = B.shape[0]
    scale_vec = np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X


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
    else:
        raise ValueError("unknown graph type")
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


@pytest.mark.xfail
def test_sanity() -> None:
    orig_dag = simulate_dag(5, 9)
    X = simulate_nonlinear_sem(orig_dag, 200)

    dag = get_dstruct_dag(X, n_iter=200, compress=False, seed=11)

    assert len(dag) > 0
    acc = count_accuracy(orig_dag, dag)

    assert acc["fdr"] < 0.4
    assert acc["tpr"] > 0.7
