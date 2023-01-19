"""
Reference: PrivBayes: Private Data Release via Bayesian Networks. (2017), Zhang J, Cormode G, Procopiuc CM, Srivastava D, Xiao X.
"""
# stdlib
from collections import namedtuple
from itertools import combinations, product
from math import ceil
from pathlib import Path
from typing import Any, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pydantic import validate_arguments
from scipy.optimize import fsolve
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# synthcity absolute
import synthcity.logger as log
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.core.serializable import Serializable
from synthcity.utils.reproducibility import enable_reproducible_results

network_edge = namedtuple("network_edge", ["feature", "parents"])


def usefulness_minus_target(
    k: int,
    num_attributes: int,
    num_tuples: int,
    target_usefulness: int = 5,
    epsilon: float = 0.1,
) -> int:
    """Usefulness function in PrivBayes.

    Parameters
    ----------
    k : int
        Max number of degree in Bayesian networks construction
    num_attributes : int
        Number of attributes in dataset.
    num_tuples : int
        Number of tuples in dataset.
    target_usefulness : int or float
    epsilon : float
        Parameter of differential privacy.
    """
    if k == num_attributes:
        usefulness = target_usefulness
    else:
        usefulness = (
            num_tuples * epsilon / ((num_attributes - k) * (2 ** (k + 3)))
        )  # PrivBayes Lemma 3
    return usefulness - target_usefulness


class PrivBayes(Serializable):
    """PrivBayes is a differentially private method for releasing high-dimensional data.

    Given a dataset D, PrivBayes first constructs a Bayesian network N , which
        (i) provides a succinct model of the correlations among the attributes in D
        (ii) allows us to approximate the distribution of data in D using a set P of lowdimensional marginals of D.

    After that, PrivBayes injects noise into each marginal in P to ensure differential privacy, and then uses the noisy marginals and the Bayesian network to construct an approximation of the data distribution in D.
    Finally, PrivBayes samples tuples from the approximate distribution to construct a synthetic dataset, and then releases the synthetic data.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        epsilon: float = 1.0,
        K: int = 0,
        n_bins: int = 100,
        mi_thresh: float = 0.01,
        target_usefulness: int = 5,
    ) -> None:
        super().__init__()
        # PrivBayes satisfies 2eps-differential privacy, eps1 + eps2 in the paper
        # eps1 = eps/2 is for the greedy bayes
        # eps2 = eps/2 is for the noisy conditionals
        self.epsilon = epsilon / 2
        self.K = K
        self.n_bins = n_bins
        self.target_usefulness = target_usefulness
        self.mi_thresh = mi_thresh
        self.default_k = 3
        self.mi_cache: dict = {}

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, data: pd.DataFrame) -> Any:
        self.n_columns = len(data.columns)
        self.n_records_fit = len(data)
        self.mi_cache = {}

        # encode dataset
        data, self.encoders = self._encode(data)

        # learn the DAG
        log.debug("[Privbayes] Run greedy Bayes")
        self.dag = self._greedy_bayes(data)
        self.ordered_nodes = []
        for attr, _ in self.dag:
            self.ordered_nodes.append(attr)

        self.display_network()

        # learn the conditional probabilities
        log.debug("[Privbayes] Compute noisy cond")
        cpds = self._compute_noisy_conditional_distributions(data)

        # create the network
        log.debug("[Privbayes] Create net")
        self.network = BayesianNetwork()

        for child, parents in self.dag:
            self.network.add_node(child)
            for parent in parents:
                self.network.add_edge(parent, child)
        self.network.add_cpds(*cpds)

        log.info(f"[PrivBayes] network is valid = {self.network.check_model()}")

        # create the model
        self.model = BayesianModelSampling(self.network)

        log.info("[PrivBayes] done training")
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def sample(self, count: int) -> pd.DataFrame:
        log.debug(f"[PrivBayes] sample {count} examples")
        samples = self.model.forward_sample(size=count, show_progress=True)

        log.debug(f"[PrivBayes] decode {count} examples")
        return self._decode(samples)

    def _encode(self, data: pd.DataFrame) -> Any:
        data = data.copy()
        encoders = {}

        for col in data.columns:
            if len(data[col].unique()) < self.n_bins or data[col].dtype.name not in [
                "object",
                "category",
            ]:
                encoders[col] = {
                    "type": "categorical",
                    "model": LabelEncoder().fit(data[col]),
                }
                data[col] = encoders[col]["model"].transform(data[col])
            else:
                col_data = pd.cut(data[col], bins=self.n_bins)
                encoders[col] = {
                    "type": "continuous",
                    "model": LabelEncoder().fit(col_data),
                }
                data[col] = encoders[col]["model"].transform(col_data)

        return data, encoders

    def _decode(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in data.columns:
            if col not in self.encoders:
                continue
            inversed = self.encoders[col]["model"].inverse_transform(data[col])
            if self.encoders[col]["type"] == "categorical":
                data[col] = inversed
            elif self.encoders[col]["type"] == "continuous":
                output = []
                for interval in inversed:
                    output.append(np.random.uniform(interval.left, interval.right))

                data[col] = output
            else:
                raise RuntimeError(f"Invalid encoder {self.encoders[col]}")

        return data

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _greedy_bayes(self, data: pd.DataFrame) -> List:
        """Construct a Bayesian Network (BN) using greedy algorithm."""
        # prepare K
        if self.K == 0:
            self.K = self._compute_K(data)

        log.info(f"[PrivBayes] Using K = {self.K}")
        # prepare data
        data.columns = data.columns.astype(str)
        num_tuples, num_attributes = data.shape

        nodes = set(data.columns)
        nodes_selected = set()

        # Init network
        network = []
        root = np.random.choice(data.columns)
        network.append(network_edge(feature=root, parents=[]))
        nodes_selected.add(root)

        nodes_remaining = nodes - nodes_selected

        for i in tqdm(range(len(nodes_remaining))):
            if len(nodes_remaining) == 0:
                break

            log.debug(f"Search node idx {i}")
            parents_pair_list = []
            mutual_info_list = []

            num_parents = min(len(nodes_selected), self.K)

            for candidate, split in product(
                nodes_remaining, range(len(nodes_selected) - num_parents + 1)
            ):
                (
                    candidate_pairs,
                    candidate_mi,
                ) = self._evaluate_parent_mutual_information(
                    data,
                    candidate=candidate,
                    parent_candidates=nodes_selected,
                    parent_limit=num_parents,
                    split=split,
                )
                parents_pair_list.extend(candidate_pairs)
                mutual_info_list.extend(candidate_mi)

            sampling_distribution = self._exponential_mechanism(
                data,
                parents_pair_list,
                mutual_info_list,
            )
            candidate_idx = np.random.choice(
                list(range(len(mutual_info_list))), p=sampling_distribution
            )
            sampled_pair = parents_pair_list[candidate_idx]
            if self.mi_thresh >= mutual_info_list[candidate_idx]:
                log.info("[PrivBayes] Weak MI score, using empty parent")
                sampled_pair = network_edge(sampled_pair.feature, parents=[])

            log.info(
                f"[PrivBayes] Sampled {sampled_pair} with score {mutual_info_list[candidate_idx]}"
            )

            nodes_selected.add(sampled_pair.feature)
            network.append(sampled_pair)

            nodes_remaining = nodes - nodes_selected
        return network

    def _laplace_noise_parameter(self, n_items: int, n_features: int) -> float:
        """The noises injected into conditional distributions.

        Note that these noises are over counts, instead of the probability distributions in PrivBayes Algorithm 1.
        """
        return 2 * (n_features - self.K) / (n_items * self.epsilon)

    def _get_noisy_counts_for_attributes(
        self, raw_data: pd.DataFrame, attributes: list
    ) -> pd.DataFrame:
        # count attribute pairs
        data = raw_data.copy().loc[:, attributes]
        data = data.sort_values(attributes)
        stats = (
            data.groupby(attributes).size().reset_index().rename(columns={0: "count"})
        )

        # add noise
        noise_para = self._laplace_noise_parameter(*raw_data.shape)
        laplace_noises = np.random.laplace(0, scale=noise_para, size=stats.index.size)

        stats["count"] += laplace_noises
        stats.loc[stats["count"] < 0, "count"] = 0

        return stats

    def _get_noisy_distribution_from_counts(
        self, stats: pd.DataFrame, attribute: str, parents: list
    ) -> pd.DataFrame:
        if len(parents) > 0:
            plist = []
            for pkey in parents:
                plist.append(stats[pkey].T)

            output = pd.crosstab(
                stats[attribute],
                plist,
                values=stats["count"],
                aggfunc="sum",
                dropna=False,
            )
            output = output.fillna(0)
            output += 1

            output = output.values
            return output / (output.sum(axis=0) + 1e-8)
        else:
            output = stats[["count"]].values
            return output / (output.sum() + 1e-8)

    def _get_noisy_distribution_for_attribute(
        self,
        data: pd.DataFrame,
        attribute: str,
        parents: list,
        counts: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        # count attribute pairs
        attributes = parents + [attribute]
        if counts is None:
            counts = self._get_noisy_counts_for_attributes(data, attributes)
        else:
            counts = counts[attributes + ["count"]]
            counts = counts.sort_values(attributes)
            counts = (
                counts.groupby(attributes)
                .sum()
                .reset_index()
                .rename(columns={0: "count"})
            )

        output = self._get_noisy_distribution_from_counts(counts, attribute, parents)

        if len(output) != data[attribute].nunique():
            raise RuntimeError(f"Invalid output len {len(output)}")
        if output.shape[1] != data[parents].nunique().prod():
            raise RuntimeError(f"Invalid output shape {output.shape}")
        return output

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _compute_noisy_conditional_distributions(
        self, data: pd.DataFrame
    ) -> np.ndarray:
        """See more in Algorithm 1 in PrivBayes."""
        conditional_distributions = []

        card = data.nunique()

        first_K_attr_counts = self._get_noisy_counts_for_attributes(
            data, self.ordered_nodes[0 : self.K]
        )
        # generate noisy conditionals for Pr[Xi | Πi] (i ∈ [0, d) ).
        for idx in range(0, len(self.dag)):
            attribute, parents = self.dag[idx]

            if idx < self.K:
                node_values = self._get_noisy_distribution_for_attribute(
                    data,
                    attribute,
                    parents,
                    counts=first_K_attr_counts,
                )  # P*(Xi | Πi)
            else:
                node_values = self._get_noisy_distribution_for_attribute(
                    data, attribute, parents
                )  # P*(Xi | Πi)

            if len(parents) == 0:
                if not np.allclose(node_values.sum().sum(), 1):
                    raise RuntimeError(f"Invalid node_values = {node_values}")
            else:
                if not np.allclose(node_values.sum(axis=0), 1):
                    raise RuntimeError(f"Invalid node_values = {node_values}")
            if not np.isnan(node_values).sum() == 0:
                raise RuntimeError(f"Invalid node_values = {node_values}")

            node_cpd = TabularCPD(
                variable=attribute,
                variable_card=card[attribute],
                values=node_values,
                evidence=parents,
                evidence_card=card[parents].values,
            )
            conditional_distributions.append(node_cpd)

        return conditional_distributions

    def _normalize_given_distribution(self, frequencies: List[float]) -> np.ndarray:
        distribution = np.array(frequencies, dtype=float)
        distribution = distribution.clip(0)  # replace negative values with 0
        summation = distribution.sum()
        if summation <= 0:
            return np.full_like(distribution, 1 / distribution.size)

        if np.isinf(summation):
            return self._normalize_given_distribution(np.isinf(distribution))
        else:
            return distribution / summation

    def _calculate_sensitivity(
        self, data: pd.DataFrame, child: str, parents: List[str]
    ) -> float:
        """Sensitivity function for Bayesian network construction. PrivBayes Lemma 4.1"""
        num_tuples = len(data)
        attr_to_is_binary = {attr: data[attr].unique().size <= 2 for attr in data}

        if attr_to_is_binary[child] or (
            len(parents) == 1 and attr_to_is_binary[parents[0]]
        ):
            a = np.log(num_tuples) / num_tuples
            b = (num_tuples - 1) / num_tuples
            b_inv = num_tuples / (num_tuples - 1)
            return a + b * np.log(b_inv)
        else:
            a = (2 / num_tuples) * np.log((num_tuples + 1) / 2)
            b = (1 - 1 / num_tuples) * np.log(1 + 2 / (num_tuples - 1))
            return a + b

    def _calculate_delta(self, data: pd.DataFrame, sensitivity: float) -> float:
        """Computing delta, which is a factor when applying differential privacy.

        More info is in PrivBayes Section 4.2 "A First-Cut Solution".
        """
        num_attributes = len(data.columns)
        return (num_attributes - 1) * sensitivity / self.epsilon

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate_parent_mutual_information(
        self,
        data: pd.DataFrame,
        candidate: str,
        parent_candidates: List[str],
        parent_limit: int,
        split: int,
    ) -> Tuple[List[network_edge], List[float]]:

        if candidate in parent_candidates:
            raise RuntimeError(f"Candidate {candidate} already in {parent_candidates}")
        if split + parent_limit > len(parent_candidates):
            return [], []

        parents_pair_list = []
        mutual_info_list = []

        if candidate not in self.mi_cache:
            self.mi_cache[candidate] = {}

        for other_parents in combinations(parent_candidates[split:], parent_limit):
            parents = list(other_parents)
            parents_key = "_".join(sorted(parents))

            if parents_key in self.mi_cache[candidate]:
                score = self.mi_cache[candidate][parents_key]
            else:
                score = self.mutual_info_score(data, parents, candidate)
                self.mi_cache[candidate][parents_key] = score

            parents_pair_list.append(network_edge(candidate, parents=parents))
            mutual_info_list.append(score)

        return parents_pair_list, mutual_info_list

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def mutual_info_score(
        self, data: pd.DataFrame, parents: List[str], candidate: str
    ) -> float:
        """Cluster the source columns, and compute the mutual information between the target and the clusters."""
        if len(parents) == 0:
            return 0

        src = data[parents]
        src_cluster = KMeans(n_clusters=10).fit(src)

        src_bins = src_cluster.predict(src)
        target = data[candidate]
        target_bins, _ = pd.cut(target, bins=self.n_bins, retbins=True)
        target_bins = LabelEncoder().fit_transform(target_bins)

        return normalized_mutual_info_score(src_bins, target_bins)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _exponential_mechanism(
        self,
        data: pd.DataFrame,
        parents_pair_list: List[network_edge],
        mutual_info_list: List[float],
    ) -> List:
        """Applied in Exponential Mechanism to sample outcomes."""
        delta_array = []
        for (candidate, parents) in parents_pair_list:
            sensitivity = self._calculate_sensitivity(data, candidate, parents)
            delta = self._calculate_delta(data, sensitivity)
            delta_array.append(delta)

        mi_array = np.array(mutual_info_list) / (2 * np.array(delta_array))
        mi_array = np.exp(mi_array)
        mi_array = self._normalize_given_distribution(mi_array)
        return mi_array

    def _compute_K(self, data: pd.DataFrame) -> int:
        """Calculate the maximum degree when constructing Bayesian networks. See PrivBayes Section 4.5."""
        num_tuples, num_attributes = data.shape

        initial_usefulness = usefulness_minus_target(
            self.default_k, num_attributes, num_tuples, 0, self.epsilon
        )
        log.info(
            f"[PrivBayes] initial_usefulness = {initial_usefulness} self.target_usefulness = {self.target_usefulness}"
        )
        if initial_usefulness > self.target_usefulness:
            return self.default_k

        arguments = (num_attributes, num_tuples, self.target_usefulness, self.epsilon)
        try:
            ans = fsolve(
                usefulness_minus_target,
                np.array([int(num_attributes / 2)]),
                args=arguments,
            )[0]
            ans = ceil(ans)
        except RuntimeWarning:
            ans = self.default_k
        if ans < 1 or ans > num_attributes:
            ans = self.default_k
        return ans

    def display_network(self) -> None:
        length = 0
        for child, _ in self.dag:
            if len(child) > length:
                length = len(child)

        log.info("Constructed Bayesian network:")
        for child, parents in self.dag:
            log.info(
                "    {0:{width}} has parents {1}.".format(child, parents, width=length)
            )


class PrivBayesPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.privacy.plugin_privbayes.PrivBayesPlugin
        :parts: 1

    PrivBayes algorithm.


        Args:
            epsilon: float
                Differential privacy parameter
            K:
                Maximum number of parents for a node
            n_bins: int
                Number of bins for encoding the features
            mi_thresh: int
                Mutual information lower threshold. If the current score is lower, the [] parents are used.
            target_usefulness: int
                Def 4.7 in the paper: A noisy distribution is θ-useful if the ratio of average scale of
    information to average scale of noise is no less than θ. 5-useful is the recommended value.
            random_state: int
                Random seed
            # Core Plugin arguments
            workspace: Path.
                Optional Path for caching intermediary results.
            compress_dataset: bool. Default = False.
                Drop redundant features before training the generator.
            sampling_patience: int.
                Max inference iterations to wait for the generated data to match the training schema.

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from synthcity.plugins import Plugins
        >>>
        >>> X, y = load_iris(as_frame = True, return_X_y = True)
        >>> X["target"] = y
        >>>
        >>> plugin = Plugins().get("privbayes")
        >>> plugin.fit(X)
        >>>
        >>> plugin.generate(50)


    """

    def __init__(
        self,
        epsilon: float = 1.0,
        K: int = 0,
        n_bins: int = 100,
        mi_thresh: float = 0.01,
        target_usefulness: int = 5,
        random_state: int = 0,
        # core plugin arguments
        workspace: Path = Path("workspace"),
        compress_dataset: bool = False,
        sampling_patience: int = 500,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            random_state=random_state,
            sampling_patience=sampling_patience,
            workspace=workspace,
            compress_dataset=compress_dataset,
            **kwargs,
        )

        enable_reproducible_results(random_state)

        self.epsilon = epsilon
        self.K = K
        self.n_bins = n_bins
        self.mi_thresh = mi_thresh
        self.target_usefulness = target_usefulness

    @staticmethod
    def name() -> str:
        return "privbayes"

    @staticmethod
    def type() -> str:
        return "privacy"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "PrivBayesPlugin":
        self.model = PrivBayes(
            epsilon=self.epsilon,
            K=self.K,
            n_bins=self.n_bins,
            mi_thresh=self.mi_thresh,
            target_usefulness=self.target_usefulness,
        )
        self.model.fit(X.dataframe())
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.sample, count, syn_schema)


plugin = PrivBayesPlugin
