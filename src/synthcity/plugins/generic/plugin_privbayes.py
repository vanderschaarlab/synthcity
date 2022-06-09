"""
Adapted from:
    - https://github.com/daanknoors/synthetic_data_generation
    - https://github.com/DataResponsibly/DataSynthesizer
"""
# stdlib
import random
from collections import namedtuple
from typing import Any, Dict, List, Set, Tuple

# third party
import numpy as np
import pandas as pd
from diffprivlib.mechanisms import Exponential
from pydantic import validate_arguments
from thomas.core import BayesianNetwork

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.utils.dp import dp_conditional_distribution
from synthcity.utils.statistics import (
    cardinality,
    compute_distribution,
    joint_distribution,
)

APPair = namedtuple("APPair", ["attribute", "parents"])


class PrivBayes:
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
        theta_usefulness: float = 4,
        epsilon_split: float = 0.3,
        score_function: str = "R",
    ) -> None:
        self.epsilon = epsilon
        self.theta_usefulness = theta_usefulness
        self.epsilon_split = epsilon_split  # also called Beta in paper
        self.score_function = score_function

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, data: pd.DataFrame) -> Any:
        self.columns_ = list(data.columns)
        self.n_records_fit_ = data.shape[0]
        self.dtypes_fit_ = data.dtypes

        self._greedy_bayes(data)
        self._compute_conditional_distributions(data)
        self.model_ = BayesianNetwork.from_CPTs("PrivBayes", self.cpt_.values())
        return self

    def sample(self, n_records: int) -> pd.DataFrame:
        return self._generate_data(n_records)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _greedy_bayes(self, data: pd.DataFrame) -> Any:
        nodes, nodes_selected = self._init_network(data)

        # normally len(nodes) - 1, unless user initialized part of the network
        self._n_nodes_dp_computed = len(nodes) - len(nodes_selected)

        for i in range(len(nodes_selected), len(nodes)):
            nodes_remaining = nodes - nodes_selected

            # select ap_pair candidates
            ap_pairs = []
            for node in nodes_remaining:
                max_domain_size = self._max_domain_size(data, node)
                max_parent_sets = self._max_parent_sets(
                    data, nodes_selected, max_domain_size
                )

                # empty set - domain size of node violates theta_usefulness
                if len(max_parent_sets) == 0 or (
                    len(max_parent_sets) == 1 and len(max_parent_sets[0]) == 1
                ):
                    ap_pairs.append(APPair(node, parents=[]))
                else:
                    ap_pairs.extend(
                        [APPair(node, parents=[p]) for p in max_parent_sets]
                    )

            scores = self._compute_scores(data, ap_pairs)
            sampled_pair = self._exponential_mechanism(ap_pairs, scores)

            nodes_selected.add(sampled_pair.attribute)
            self.network_.append(sampled_pair)
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _max_domain_size(self, data: pd.DataFrame, node: str) -> int:
        """Computes the maximum domain size a node can have to satisfy theta-usefulness"""
        node_cardinality = cardinality(data[node])
        max_domain_size = (
            self.n_records_fit_ * (1 - self.epsilon_split) * self.epsilon
        ) / (2 * len(self.columns_) * self.theta_usefulness * node_cardinality)
        return max_domain_size

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _max_parent_sets(
        self, data: pd.DataFrame, v: Set, max_domain_size: float
    ) -> List[Set]:
        """Refer to algorithm 5 in paper - max parent set is 1) theta-useful and 2) maximal."""
        if max_domain_size < 1:
            return []
        if len(v) == 0:
            return []

        x = np.random.choice(tuple(v))
        x_domain_size = cardinality(data[x])
        x = {x}

        v_without_x = v - x

        parent_sets1 = self._max_parent_sets(data, v_without_x, max_domain_size)
        parent_sets2 = self._max_parent_sets(
            data, v_without_x, max_domain_size / x_domain_size
        )

        for z in parent_sets2:
            if z in parent_sets1:
                parent_sets1.remove(z)
            parent_sets1.append(z.union(x))
        return parent_sets1

    def _init_network(self, X: pd.DataFrame) -> Tuple[Set, Set]:
        self._binary_columns = [c for c in X.columns if X[c].unique().size <= 2]
        nodes = set(X.columns)

        # if set_network is not called we start with a random first node
        self.network_ = []
        nodes_selected = set()

        root = np.random.choice(tuple(nodes))
        self.network_.append(APPair(attribute=root, parents=[]))
        nodes_selected.add(root)
        return nodes, nodes_selected

    def _compute_scores(self, data: pd.DataFrame, ap_pairs: list) -> list:
        """Compute score for all ap_pairs"""
        return [
            self.mi_score(data, [pair.attribute], pair.parents) for pair in ap_pairs
        ]

    def _score_sensitivity(self) -> float:
        """Checks input score function and sets sensitivity"""
        if self.score_function.upper() not in ["R", "MI"]:
            raise ValueError("Score function must be 'R' or 'MI'")

        if self.score_function.upper() == "R":
            return (3 / self.n_records_fit_) + (2 / self.n_records_fit_**2)

        # note: for simplicity we assume that all APPairs are non-binary, which is the upperbound of MI sensitivity
        elif self.score_function.upper() == "MI":
            return (2 / self.n_records_fit_) * np.log((self.n_records_fit_ + 1) / 2) + (
                ((self.n_records_fit_ - 1) / self.n_records_fit_)
                * np.log((self.n_records_fit_ + 1) / (self.n_records_fit_ - 1))
            )

        raise RuntimeError(f"Invalid score function {self.score_function}")

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _exponential_mechanism(self, ap_pairs: list, scores: list) -> APPair:
        """select APPair with exponential mechanism"""
        local_epsilon = self.epsilon * self.epsilon_split / self._n_nodes_dp_computed
        dp_mech = Exponential(
            epsilon=local_epsilon,
            sensitivity=self._score_sensitivity(),
            utility=list(scores),
            candidates=ap_pairs,
        )
        sampled_pair = dp_mech.randomise()
        return sampled_pair

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _compute_conditional_distributions(self, data: pd.DataFrame) -> Any:
        self.cpt_ = dict()

        local_epsilon = self.epsilon * (1 - self.epsilon_split) / len(self.columns_)

        for idx, pair in enumerate(self.network_):
            if len(pair.parents) == 0:
                attributes = [pair.attribute]
            else:
                attributes = [*pair.parents, pair.attribute]

            dp_cpt = dp_conditional_distribution(
                data[attributes], epsilon=local_epsilon
            )
            self.cpt_[pair.attribute] = dp_cpt
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _generate_data(self, n_records: int) -> np.ndarray:
        data_synth = np.empty([n_records, len(self.columns_)], dtype=object)

        for i in range(n_records):
            record = self._sample_record()
            data_synth[i] = list(record.values())

        # numpy.array to pandas.DataFrame with original column ordering
        data_synth = pd.DataFrame(
            data_synth, columns=[c.attribute for c in self.network_]
        )
        return data_synth

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _sample_record(self) -> Dict:
        """samples a value column for column by conditioning for parents"""
        record: Dict[str, list] = {}
        for col_idx, pair in enumerate(self.network_):
            node = self.model_[pair.attribute]
            node_cpt = node.cpt
            node_states = node.states

            if node.conditioning:
                parent_values = [record[p] for p in node.conditioning]
                node_probs = node_cpt[tuple(parent_values)]

            else:
                node_probs = node_cpt.values
            # use random.choices over np.random.choice as np coerces, e.g. sample['nan', 1, 3.0] -> '1' (int to string)
            sampled_node_value = random.choices(node_states, weights=node_probs, k=1)[
                0
            ]  # returns list

            record[node.name] = sampled_node_value
        return record

    @staticmethod
    def mi_score(data: pd.DataFrame, columns_a: list, columns_b: list) -> float:
        if len(columns_b) == 0 or len(columns_a) == 0:
            return 0
        prob_a = compute_distribution(data[columns_a])
        prob_b = compute_distribution(data[columns_b])
        prob_joint = compute_distribution(data[columns_a + columns_b])

        # todo: pull-request thomas to add option to normalize to remove 0's
        # align
        prob_div = prob_joint / (prob_b * prob_a)
        prob_joint, prob_div = prob_joint.extend_and_reorder(prob_joint, prob_div)

        # remove zeros as this will result in issues with log
        prob_joint = prob_joint.values[prob_joint.values != 0]
        prob_div = prob_div.values[prob_div.values != 0]
        mi = np.sum(prob_joint * np.log(prob_div))
        # mi = np.sum(p_nodeparents.values * np.log(p_nodeparents / (p_parents * p_node)))
        return mi

    @staticmethod
    def r_score(data: pd.DataFrame, columns_a: list, columns_b: list) -> float:
        """An alternative score function to mutual information with lower sensitivity - can be used on non-binary domains.
        Relies on the L1 distance from a joint distribution to a joint distributions that minimizes mutual information.
        Refer to Lemma 5.2
        """
        if len(columns_b) == 0:
            return 0
        # compute distribution that minimizes mutual information
        prob_a = compute_distribution(data[columns_a])
        prob_b = compute_distribution(data[columns_b])
        prob_independent = prob_b * prob_a

        # compute joint distribution
        prob_joint = joint_distribution(data[columns_a + columns_b])

        # substract not part of thomas - need to ensure alignment
        prob_joint, prob_independent = prob_joint.extend_and_reorder(
            prob_joint, prob_independent
        )
        l1_distance = 0.5 * np.sum(np.abs(prob_joint.values - prob_independent.values))
        return l1_distance


class PrivBayesPlugin(Plugin):
    """PrivBayes algorithm.

    Paper: PrivBayes: Private Data Release via Bayesian Networks. (2017), Zhang J, Cormode G, Procopiuc CM, Srivastava D, Xiao X."""

    def __init__(
        self,
        dp_epsilon: float = 1.0,
        theta_usefulness: float = 4,
        epsilon_split: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.dp_epsilon = dp_epsilon
        self.theta_usefulness = theta_usefulness
        self.epsilon_split = epsilon_split

    @staticmethod
    def name() -> str:
        return "privbayes"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "PrivBayesPlugin":
        self.model = PrivBayes(
            epsilon=self.dp_epsilon,
            theta_usefulness=self.theta_usefulness,
            epsilon_split=self.epsilon_split,
            score_function="R",
        )
        self.model.fit(X.dataframe())
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.sample, count, syn_schema)


plugin = PrivBayesPlugin
