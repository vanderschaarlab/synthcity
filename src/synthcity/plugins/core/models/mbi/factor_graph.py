# stdlib
from collections import defaultdict

# third party
import numpy as np

# synthcity relative
from .clique_vector import CliqueVector
from .factor import Factor


class FactorGraph:
    def __init__(self, domain, cliques, total=1.0, convex=False, iters=25):
        self.domain = domain
        self.cliques = cliques
        self.total = total
        self.convex = convex
        self.iters = iters

        if convex:
            self.counting_numbers = self.get_counting_numbers()
            self.belief_propagation = self.convergent_belief_propagation
        else:
            counting_numbers = {}
            for cl in cliques:
                counting_numbers[cl] = 1.0
            for a in domain:
                counting_numbers[a] = 1.0 - len([cl for cl in cliques if a in cl])
            self.counting_numbers = None, None, counting_numbers
            self.belief_propagation = self.loopy_belief_propagation

        self.potentials = None
        self.marginals = None
        self.messages = self.init_messages()
        self.beliefs = {i: Factor.zeros(domain.project(i)) for i in domain}

    def datavector(self, flatten=True):
        """Materialize the explicit representation of the distribution as a data vector."""
        logp = sum(self.potentials[cl] for cl in self.cliques)
        ans = np.exp(logp - logp.logsumexp())
        wgt = ans.domain.size() / self.domain.size()
        return ans.expand(self.domain).datavector(flatten) * wgt * self.total

    def init_messages(self):
        mu_n = defaultdict(dict)
        mu_f = defaultdict(dict)
        for cl in self.cliques:
            for v in cl:
                mu_f[cl][v] = Factor.zeros(self.domain.project(v))
                mu_n[v][cl] = Factor.zeros(self.domain.project(v))
        return mu_n, mu_f

    def primal_feasibility(self, mu):
        ans = 0
        count = 0
        for r in mu:
            for s in mu:
                if r == s:
                    break
                d = tuple(set(r) & set(s))
                if len(d) > 0:
                    x = mu[r].project(d).datavector()
                    y = mu[s].project(d).datavector()
                    err = np.linalg.norm(x - y, 1)
                    ans += err
                    count += 1
        try:
            return ans / count
        except BaseException:
            return 0

    def project(self, attrs):
        if type(attrs) is list:
            attrs = tuple(attrs)

        if self.marginals is not None:
            # we will average all ways to obtain the given marginal,
            # since there may be more than one
            ans = Factor.zeros(self.domain.project(attrs))
            terminate = False
            for cl in self.cliques:
                if set(attrs) <= set(cl):
                    ans += self.marginals[cl].project(attrs)
                    terminate = True
            if terminate:
                return ans * (self.total / ans.sum())

        belief = sum(self.beliefs[i] for i in attrs)
        belief += np.log(self.total) - belief.logsumexp()
        return belief.transpose(attrs).exp()

    def loopy_belief_propagation(self, potentials, callback=None):
        mu_n, mu_f = self.messages
        self.potentials = potentials

        for i in range(self.iters):
            # factor to variable BP
            for cl in self.cliques:
                pre = sum(mu_n[c][cl] for c in cl)
                for v in cl:
                    complement = [var for var in cl if var is not v]
                    mu_f[cl][v] = potentials[cl] + pre - mu_n[v][cl]
                    mu_f[cl][v] = mu_f[cl][v].logsumexp(complement)
                    mu_f[cl][v] -= mu_f[cl][v].logsumexp()

            # variable to factor BP
            for v in self.domain:
                fac = [cl for cl in self.cliques if v in cl]
                pre = sum(mu_f[cl][v] for cl in fac)
                for f in fac:
                    complement = [var for var in fac if var is not f]
                    # mu_n[v][f] = Factor.zeros(self.domain.project(v))
                    mu_n[v][f] = pre - mu_f[f][v]  # sum(mu_f[c][v] for c in complement)
                    # mu_n[v][f] += sum(mu_f[c][v] for c in complement)
                    # mu_n[v][f] -= mu_n[v][f].logsumexp()

            if callback is not None:
                mg = self.clique_marginals(mu_n, mu_f, potentials)
                callback(mg)

        self.beliefs = {
            v: sum(mu_f[cl][v] for cl in self.cliques if v in cl) for v in self.domain
        }
        self.messages = mu_n, mu_f
        self.marginals = self.clique_marginals(mu_n, mu_f, potentials)
        return self.marginals

    def convergent_belief_propagation(self, potentials, callback=None):
        # Algorithm 11.2 in Koller & Friedman (modified to work in log space)

        v, vhat, k = self.counting_numbers
        sigma, delta = self.messages
        # sigma, delta = self.init_messages()

        for it in range(self.iters):
            # pre = {}
            # for r in self.cliques:
            #    pre[r] = sum(sigma[j][r] for j in r)

            for i in self.domain:
                nbrs = [r for r in self.cliques if i in r]
                for r in nbrs:
                    comp = [j for j in r if i != j]
                    delta[r][i] = potentials[r] + sum(sigma[j][r] for j in comp)
                    # delta[r][i] = potentials[r] + pre[r] - sigma[i][r]
                    delta[r][i] /= vhat[i, r]
                    delta[r][i] = delta[r][i].logsumexp(comp)
                belief = Factor.zeros(self.domain.project(i))
                belief += sum(delta[r][i] * vhat[i, r] for r in nbrs) / vhat[i]
                belief -= belief.logsumexp()
                self.beliefs[i] = belief
                for r in nbrs:
                    comp = [j for j in r if i != j]
                    A = -v[i, r] / vhat[i, r]
                    B = v[r]
                    sigma[i][r] = A * (potentials[r] + sum(sigma[j][r] for j in comp))
                    # sigma[i][r] = A*(potentials[r] + pre[r] - sigma[i][r])
                    sigma[i][r] += B * (belief - delta[r][i])
            if callback is not None:
                mg = self.clique_marginals(sigma, delta, potentials)
                callback(mg)

        self.messages = sigma, delta
        return self.clique_marginals(sigma, delta, potentials)

    def clique_marginals(self, mu_n, mu_f, potentials):
        if self.convex:
            v, _, _ = self.counting_numbers
        marginals = {}
        for cl in self.cliques:
            belief = potentials[cl] + sum(mu_n[n][cl] for n in cl)
            if self.convex:
                belief *= 1.0 / v[cl]
            belief += np.log(self.total) - belief.logsumexp()
            marginals[cl] = belief.exp()
        return CliqueVector(marginals)

    def mle(self, marginals):
        return -self.bethe_entropy(marginals)[1]

    def bethe_entropy(self, marginals):
        """
        Return the Bethe Entropy and the gradient with respect to the marginals

        """
        _, _, weights = self.counting_numbers
        entropy = 0
        dmarginals = {}
        attributes = set()
        for cl in self.cliques:
            mu = marginals[cl] / self.total
            entropy += weights[cl] * (mu * mu.log()).sum()
            dmarginals[cl] = weights[cl] * (1 + mu.log()) / self.total
            for a in set(cl) - set(attributes):
                p = mu.project(a)
                entropy += weights[a] * (p * p.log()).sum()
                dmarginals[cl] += weights[a] * (1 + p.log()) / self.total
                attributes.update(a)

        return -entropy, -1 * CliqueVector(dmarginals)

    def get_counting_numbers(self):
        # third party
        from cvxopt import matrix, solvers

        solvers.options["show_progress"] = False
        index = {}
        idx = 0

        for i in self.domain:
            index[i] = idx
            idx += 1
        for r in self.cliques:
            index[r] = idx
            idx += 1

        for r in self.cliques:
            for i in r:
                index[r, i] = idx
                idx += 1

        vectors = {}
        for r in self.cliques:
            v = np.zeros(idx)
            v[index[r]] = 1
            for i in r:
                v[index[r, i]] = 1
            vectors[r] = v

        for i in self.domain:
            v = np.zeros(idx)
            v[index[i]] = 1
            for r in self.cliques:
                if i in r:
                    v[index[r, i]] = -1
            vectors[i] = v

        constraints = []
        for i in self.domain:
            con = vectors[i].copy()
            for r in self.cliques:
                if i in r:
                    con += vectors[r]
            constraints.append(con)
        A = np.array(constraints)
        b = np.ones(len(self.domain))

        X = np.vstack([vectors[r] for r in self.cliques])
        y = np.ones(len(self.cliques))
        P = X.T @ X
        q = -X.T @ y
        G = -np.eye(q.size)
        h = np.zeros(q.size)
        minBound = 1.0 / len(self.domain)
        for r in self.cliques:
            h[index[r]] = -minBound

        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)

        ans = solvers.qp(P, q, G, h, A, b)
        x = np.array(ans["x"]).flatten()

        counting_v = {}
        for r in self.cliques:
            counting_v[r] = x[index[r]]
            for i in r:
                counting_v[i, r] = x[index[r, i]]
        for i in self.domain:
            counting_v[i] = x[index[i]]

        counting_vhat = {}
        counting_k = {}
        for i in self.domain:
            nbrs = [r for r in self.cliques if i in r]
            counting_vhat[i] = counting_v[i] + sum(counting_v[r] for r in nbrs)
            counting_k[i] = counting_v[i] - sum(counting_v[i, r] for r in nbrs)
            for r in nbrs:
                counting_vhat[i, r] = counting_v[r] + counting_v[i, r]
        for r in self.cliques:
            counting_k[r] = counting_v[r] + sum(counting_v[i, r] for i in r)

        return counting_v, counting_vhat, counting_k
