# stdlib
import itertools
from functools import reduce

# third party
import cloudpickle
import networkx as nx
import numpy as np
import pandas as pd

# synthcity relative
from .clique_vector import CliqueVector
from .dataset import Dataset
from .domain import Domain
from .junction_tree import JunctionTree


class GraphicalModel:
    def __init__(self, domain, cliques, total=1.0, elimination_order=None):
        """Constructor for a GraphicalModel

        :param domain: a Domain object
        :param total: the normalization constant for the distribution
        :param cliques: a list of cliques (not necessarilly maximal cliques)
            - each clique is a subset of attributes, represented as a tuple or list
        :param elim_order: an elimination order for the JunctionTree algorithm
            - Elimination order will impact the efficiency by not correctness.
              By default, a greedy elimination order is used
        """
        self.domain = domain
        self.total = total
        tree = JunctionTree(domain, cliques, elimination_order)
        self.junction_tree = tree

        self.cliques = tree.maximal_cliques()  # maximal cliques
        self.message_order = tree.mp_order()
        self.sep_axes = tree.separator_axes()
        self.neighbors = tree.neighbors()
        self.elimination_order = tree.elimination_order

        self.size = sum(domain.size(cl) for cl in self.cliques)
        if self.size * 8 > 4 * 10**9:
            # stdlib
            import warnings

            message = "Size of parameter vector is %.2f GB. " % (
                self.size * 8 / 10**9
            )
            message += "Consider removing some measurements or finding a better elimination order"
            warnings.warn(message)

    @staticmethod
    def save(model, path):
        cloudpickle.dump(model, open(path, "wb"))

    @staticmethod
    def load(path):
        return cloudpickle.load(open(path, "rb"))

    def project(self, attrs):
        """Project the distribution onto a subset of attributes.
            I.e., compute the marginal of the distribution

        :param attrs: a subset of attributes in the domain, represented as a list or tuple
        :return: a Factor object representing the marginal distribution
        """
        # use precalculated marginals if possible
        if type(attrs) is list:
            attrs = tuple(attrs)
        if hasattr(self, "marginals"):
            for cl in self.cliques:
                if set(attrs) <= set(cl):
                    return self.marginals[cl].project(attrs)

        elim = self.domain.invert(attrs)
        elim_order = greedy_order(self.domain, self.cliques + [attrs], elim)
        pots = list(self.potentials.values())
        ans = variable_elimination_logspace(pots, elim_order, self.total)
        return ans.project(attrs)

    def krondot(self, matrices):
        """Compute the answer to the set of queries Q1 x Q2 X ... x Qd, where
            Qi is a query matrix on the ith attribute and "x" is the Kronecker product
        This may be more efficient than computing a supporting marginal then multiplying that by Q.
        In particular, if each Qi has only a few rows.

        :param matrices: a list of matrices for each attribute in the domain
        :return: the vector of query answers
        """
        if any(M.shape[1] != n for M, n in zip(matrices, self.domain.shape)):
            raise ValueError("matrices must conform to the shape of the domain")
        logZ = self.belief_propagation(self.potentials, logZ=True)
        factors = [self.potentials[cl].exp() for cl in self.cliques]
        Factor = type(factors[0])  # infer the type of the factors
        elim = self.domain.attrs
        for attr, Q in zip(elim, matrices):
            d = Domain(["%s-answer" % attr, attr], Q.shape)
            factors.append(Factor(d, Q))
        result = variable_elimination(factors, elim)
        result = result.transpose(["%s-answer" % a for a in elim])
        return result.datavector(flatten=False) * self.total / np.exp(logZ)

    def calculate_many_marginals(self, projections):
        """Calculates marginals for all the projections in the list using
            Algorithm for answering many out-of-clique queries (section 10.3 in Koller and Friedman)

        This method may be faster than calling project many times

        :param projections: a list of projections, where
            each projection is a subset of attributes (represented as a list or tuple)
        :return: a list of marginals, where each marginal is represented as a Factor
        """

        self.marginals = self.belief_propagation(self.potentials)
        sep = self.sep_axes
        neighbors = self.neighbors
        # first calculate P(Cj | Ci) for all neighbors Ci, Cj
        conditional = {}
        for Ci in neighbors:
            for Cj in neighbors[Ci]:
                Sij = sep[(Cj, Ci)]
                Z = self.marginals[Cj]
                conditional[(Cj, Ci)] = Z / Z.project(Sij)

        # now iterate through pairs of cliques in order of distance
        pred, dist = nx.floyd_warshall_predecessor_and_distance(
            self.junction_tree.tree, weight=False
        )
        results = {}
        for Ci, Cj in sorted(
            itertools.combinations(self.cliques, 2), key=lambda X: dist[X[0]][X[1]]
        ):
            Cl = pred[Ci][Cj]
            Y = conditional[(Cj, Cl)]
            if Cl == Ci:
                X = self.marginals[Ci]
                results[(Ci, Cj)] = results[(Cj, Ci)] = X * Y
            else:
                X = results[(Ci, Cl)]
                S = set(Cl) - set(Ci) - set(Cj)
                results[(Ci, Cj)] = results[(Cj, Ci)] = (X * Y).sum(S)

        results = {
            self.domain.canonical(key[0] + key[1]): results[key] for key in results
        }

        answers = {}
        for proj in projections:
            for attr in results:
                if set(proj) <= set(attr):
                    answers[proj] = results[attr].project(proj)
                    break
            if proj not in answers:
                # just use variable elimination
                answers[proj] = self.project(proj)

        return answers

    def datavector(self, flatten=True):
        """Materialize the explicit representation of the distribution as a data vector."""
        logp = sum(self.potentials[cl] for cl in self.cliques)
        ans = np.exp(logp - logp.logsumexp())
        wgt = ans.domain.size() / self.domain.size()
        return ans.expand(self.domain).datavector(flatten) * wgt * self.total

    def belief_propagation(self, potentials, logZ=False):
        """Compute the marginals of the graphical model with given parameters

        Note this is an efficient, numerically stable implementation of belief propagation

        :param potentials: the (log-space) parameters of the graphical model
        :param logZ: flag to return logZ instead of marginals
        :return marginals: the marginals of the graphical model
        """
        beliefs = {cl: potentials[cl].copy() for cl in potentials}
        messages = {}
        for i, j in self.message_order:
            sep = beliefs[i].domain.invert(self.sep_axes[(i, j)])
            if (j, i) in messages:
                tau = beliefs[i] - messages[(j, i)]
            else:
                tau = beliefs[i]
            messages[(i, j)] = tau.logsumexp(sep)
            beliefs[j] += messages[(i, j)]

        cl = self.cliques[0]
        if logZ:
            return beliefs[cl].logsumexp()

        logZ = beliefs[cl].logsumexp()
        for cl in self.cliques:
            beliefs[cl] += np.log(self.total) - logZ
            beliefs[cl] = beliefs[cl].exp(out=beliefs[cl])

        return CliqueVector(beliefs)

    def mle(self, marginals):
        """Compute the model parameters from the given marginals

        :param marginals: target marginals of the distribution
        :param: the potentials of the graphical model with the given marginals
        """
        potentials = {}
        variables = set()
        for cl in self.cliques:
            new = tuple(variables & set(cl))
            # factor = marginals[cl] / marginals[cl].project(new)
            variables.update(cl)
            potentials[cl] = marginals[cl].log() - marginals[cl].project(new).log()
        return CliqueVector(potentials)

    def fit(self, data):
        # synthcity relative
        from .factor import Factor

        if not data.domain.contains(self.domain):
            raise ValueError("data domain not compatible with model domain")
        marginals = {}
        for cl in self.cliques:
            x = data.project(cl).datavector()
            dom = self.domain.project(cl)
            marginals[cl] = Factor(dom, x)
        self.potentials = self.mle(marginals)

    def synthetic_data(self, rows=None, method="round"):
        """Generate synthetic tabular data from the distribution.
        Valid options for method are 'round' and 'sample'."""
        total = int(self.total) if rows is None else rows
        cols = self.domain.attrs
        data = np.zeros((total, len(cols)), dtype=int)
        df = pd.DataFrame(data, columns=cols)
        cliques = [set(cl) for cl in self.cliques]

        def synthetic_col(counts, total):
            if method == "sample":
                probas = counts / counts.sum()
                return np.random.choice(counts.size, total, True, probas)
            counts *= total / counts.sum()
            frac, integ = np.modf(counts)
            integ = integ.astype(int)
            extra = total - integ.sum()
            if extra > 0:
                idx = np.random.choice(counts.size, extra, False, frac / frac.sum())
                integ[idx] += 1
            vals = np.repeat(np.arange(counts.size), integ)
            np.random.shuffle(vals)
            return vals

        order = self.elimination_order[::-1]
        col = order[0]
        marg = self.project([col]).datavector(flatten=False)
        df.loc[:, col] = synthetic_col(marg, total)
        used = {col}

        for col in order[1:]:
            relevant = [cl for cl in cliques if col in cl]
            relevant = used.intersection(set.union(*relevant))
            proj = tuple(relevant)
            used.add(col)
            marg = self.project(proj + (col,)).datavector(flatten=False)

            def foo(group):
                idx = group.name
                vals = synthetic_col(marg[idx], group.shape[0])
                group[col] = vals
                return group

            if len(proj) >= 1:
                df = df.groupby(list(proj), group_keys=False).apply(foo)
            else:
                df[col] = synthetic_col(marg, df.shape[0])

        return Dataset(df, self.domain)


def variable_elimination_logspace(potentials, elim, total):
    """run variable elimination on a list of **logspace** factors"""
    k = len(potentials)
    psi = dict(zip(range(k), potentials))
    for z in elim:
        psi2 = [psi.pop(i) for i in list(psi.keys()) if z in psi[i].domain]
        phi = reduce(lambda x, y: x + y, psi2, 0)
        tau = phi.logsumexp([z])
        psi[k] = tau
        k += 1
    ans = reduce(lambda x, y: x + y, psi.values(), 0)
    return (ans - ans.logsumexp() + np.log(total)).exp()


def variable_elimination(factors, elim):
    """run variable elimination on a list of (non-logspace) factors"""
    k = len(factors)
    psi = dict(zip(range(k), factors))
    for z in elim:
        psi2 = [psi.pop(i) for i in list(psi.keys()) if z in psi[i].domain]
        phi = reduce(lambda x, y: x * y, psi2, 1)
        tau = phi.sum([z])
        psi[k] = tau
        k += 1
    return reduce(lambda x, y: x * y, psi.values(), 1)


def greedy_order(domain, cliques, elim):
    order = []
    unmarked = set(elim)
    cliques = set(cliques)
    total_cost = 0
    for k in range(len(elim)):
        cost = {}
        for a in unmarked:
            # all cliques that have a
            neighbors = list(filter(lambda cl: a in cl, cliques))
            # variables in this "super-clique"
            variables = tuple(set.union(set(), *map(set, neighbors)))
            # domain for the resulting factor
            newdom = domain.project(variables)
            # cost of removing a
            cost[a] = newdom.size()

        # find the best variable to eliminate
        a = min(cost, key=lambda a: cost[a])

        # do some cleanup
        order.append(a)
        unmarked.remove(a)
        neighbors = list(filter(lambda cl: a in cl, cliques))
        variables = tuple(set.union(set(), *map(set, neighbors)) - {a})
        cliques -= set(neighbors)
        cliques.add(variables)
        total_cost += cost[a]

    return order
