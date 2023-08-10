# stdlib
from collections import defaultdict
from copy import deepcopy

# third party
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr

# synthcity absolute
from synthcity import logger

# synthcity relative
from . import callbacks
from .clique_vector import CliqueVector
from .factor_graph import FactorGraph
from .region_graph import RegionGraph

"""
This file implements Approx-Private-PGM from the following paper:

Relaxed Marginal Consistency for Differentially Private Query Answering
https://arxiv.org/pdf/2109.06153.pdf
"""


class LocalInference:
    def __init__(
        self,
        domain,
        backend="numpy",
        structural_zeros={},
        metric="L2",
        log=False,
        iters=1000,
        warm_start=False,
        marginal_oracle="convex",
        inner_iters=1,
    ):
        """
        Class for learning a GraphicalModel from  noisy measurements on a data distribution

        :param domain: The domain information (A Domain object)
        :param backend: numpy or torch backend
        :param structural_zeros: An encoding of the known (structural) zeros in the distribution.
            Specified as a dictionary where
                - each key is a subset of attributes of size r
                - each value is a list of r-tuples corresponding to impossible attribute settings
        :param metric: The optimization metric.  May be L1, L2 or a custom callable function
            - custom callable function must consume the marginals and produce the loss and gradient
            - see FactoredInference._marginal_loss for more information
        :param log: flag to log iterations of optimization
        :param iters: number of iterations to optimize for
        :param warm_start: initialize new model or reuse last model when calling infer multiple times
        :param marginal_oracle: One of
            - convex (Region graph, convex Kikuchi entropy)
            - approx (Region graph, Kikuchi entropy)
            - pairwise-convex (Factor graph, convex Bethe entropy)
            - pairwise (Factor graph, Bethe entropy)
            - Can also pass any and FactorGraph or RegionGraph object
        """
        self.domain = domain
        self.backend = backend
        self.metric = metric
        self.log = log
        self.iters = iters
        self.warm_start = warm_start
        self.history = []
        self.marginal_oracle = marginal_oracle
        self.inner_iters = inner_iters
        if backend == "torch":
            # third party
            from mbi.torch_factor import Factor

            self.Factor = Factor
        else:
            # third party
            from mbi import Factor

            self.Factor = Factor

        self.structural_zeros = CliqueVector({})
        for cl in structural_zeros:
            dom = self.domain.project(cl)
            fact = structural_zeros[cl]
            self.structural_zeros[cl] = self.Factor.active(dom, fact)

    def estimate(self, measurements, total=None, callback=None, options={}):
        """
        Estimate a GraphicalModel from the given measurements

        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param callback: a function to be called after each iteration of optimization
        :param options: solver specific options passed as a dictionary
            { param_name : param_value }

        :return model: A GraphicalModel that best matches the measurements taken
        """
        options["callback"] = callback
        if callback is None and self.log:
            options["callback"] = callbacks.Logger(self)
        self.mirror_descent(measurements, total, **options)
        return self.model

    def mirror_descent_auto(self, alpha, iters, callback=None):
        model = self.model
        theta0 = model.potentials
        messages0 = deepcopy(model.messages)
        theta = theta0
        mu = model.belief_propagation(theta)
        l0, _ = self._marginal_loss(mu)

        prev_l = np.inf
        for t in range(iters):
            if callback is not None:
                callback(mu)
            l, dL = self._marginal_loss(mu)
            theta = theta - alpha * dL
            mu = model.belief_propagation(theta)
            if l > prev_l:
                if t <= 50:
                    if self.log:
                        logger.debug(
                            f"Reducing learning rate and restarting. alpha/2: {alpha / 2}"
                        )
                    model.potentials = theta0
                    model.messages = messages0
                    return self.mirror_descent_auto(alpha / 2, iters, callback)
                else:
                    model.damping = (0.9 + model.damping) / 2.0
                    if self.log:
                        logger.debug(
                            f"Increasing damping and continuing. Damping: {model.damping}"
                        )
                    alpha *= 0.5
            prev_l = l

        # run some extra iterations with no gradient update to make sure things are primal feasible
        for _ in range(1000):
            if model.primal_feasibility(mu) < 1.0:
                break
            mu = model.belief_propagation(theta)
            if callback is not None:
                callback(mu)
        return l, theta, mu

    def mirror_descent(
        self, measurements, total=None, initial_alpha=10.0, callback=None
    ):
        """Use the mirror descent algorithm to estimate the GraphicalModel
            See https://web.iem.technion.ac.il/images/user-files/becka/papers/3.pdf

        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param stepsize: the learning rate function
        :param callback: a function to be called after each iteration of optimization
        """
        self._setup(measurements, total)
        l, theta, mu = self.mirror_descent_auto(
            alpha=initial_alpha, iters=self.iters, callback=callback
        )

        self.model.potentials = theta
        self.model.marginals = mu

        return l

    def _marginal_loss(self, marginals, metric=None):
        """Compute the loss and gradient for a given dictionary of marginals

        :param marginals: A dictionary with keys as projections and values as Factors
        :return loss: the loss value
        :return grad: A dictionary with gradient for each marginal
        """
        if metric is None:
            metric = self.metric

        if callable(metric):
            return metric(marginals)

        loss = 0.0
        gradient = {}

        for cl in marginals:
            mu = marginals[cl]
            gradient[cl] = self.Factor.zeros(mu.domain)
            for Q, y, noise, proj in self.groups[cl]:
                c = 1.0 / noise
                mu2 = mu.project(proj)
                x = mu2.datavector()
                diff = c * (Q @ x - y)
                if metric == "L1":
                    loss += abs(diff).sum()
                    sign = diff.sign() if hasattr(diff, "sign") else np.sign(diff)
                    grad = c * (Q.T @ sign)
                else:
                    loss += 0.5 * (diff @ diff)
                    grad = c * (Q.T @ diff)
                gradient[cl] += self.Factor(mu2.domain, grad)
        return float(loss), CliqueVector(gradient)

    def _setup(self, measurements, total):
        """Perform necessary setup for running estimation algorithms

        1. If total is None, find the minimum variance unbiased estimate for total and use that
        2. Construct the GraphicalModel
            * If there are structural_zeros in the distribution, initialize factors appropriately
        3. Pre-process measurements into groups so that _marginal_loss may be evaluated efficiently
        """
        if total is None:
            # find the minimum variance estimate of the total given the measurements
            variances = np.array([])
            estimates = np.array([])
            for Q, y, noise, proj in measurements:
                o = np.ones(Q.shape[1])
                v = lsmr(Q.T, o, atol=0, btol=0)[0]
                if np.allclose(Q.T.dot(v), o):
                    variances = np.append(variances, noise**2 * np.dot(v, v))
                    estimates = np.append(estimates, np.dot(v, y))
            if estimates.size == 0:
                total = 1
            else:
                variance = 1.0 / np.sum(1.0 / variances)
                estimate = variance * np.sum(estimates / variances)
                total = max(1, estimate)

        # if not self.warm_start or not hasattr(self, 'model'):
        # initialize the model and parameters
        cliques = [m[3] for m in measurements]
        if self.structural_zeros is not None:
            cliques += list(self.structural_zeros.keys())
        if self.marginal_oracle == "approx":
            model = RegionGraph(
                self.domain, cliques, total, convex=False, iters=self.inner_iters
            )
        elif self.marginal_oracle == "convex":
            model = RegionGraph(
                self.domain, cliques, total, convex=True, iters=self.inner_iters
            )
        elif self.marginal_oracle == "pairwise":
            model = FactorGraph(
                self.domain, cliques, total, convex=False, iters=self.inner_iters
            )
        elif self.marginal_oracle == "pairwise-convex":
            model = FactorGraph(
                self.domain, cliques, total, convex=True, iters=self.inner_iters
            )
        else:
            model = self.marginal_oracle
            model.total = total

        if type(self.marginal_oracle) is str:
            model.potentials = CliqueVector.zeros(self.domain, model.cliques)
            model.potentials.combine(self.structural_zeros)
            if self.warm_start and hasattr(self, "model"):
                model.potentials.combine(self.model.potentials)
        self.model = model

        # group the measurements into model cliques
        cliques = self.model.cliques
        # self.groups = { cl : [] for cl in cliques }
        self.groups = defaultdict(lambda: [])
        for Q, y, noise, proj in measurements:
            if self.backend == "torch":
                # third party
                import torch

                device = self.Factor.device
                y = torch.tensor(y, dtype=torch.float32, device=device)
                if isinstance(Q, np.ndarray):
                    Q = torch.tensor(Q, dtype=torch.float32, device=device)
                elif sparse.issparse(Q):
                    Q = Q.tocoo()
                    idx = torch.LongTensor([Q.row, Q.col])
                    vals = torch.FloatTensor(Q.data)
                    Q = torch.sparse.FloatTensor(idx, vals).to(device)

                # else Q is a Linear Operator, must be compatible with torch
            m = (Q, y, noise, proj)
            for cl in sorted(cliques, key=model.domain.size):
                # (Q, y, noise, proj) tuple
                if set(proj) <= set(cl):
                    self.groups[cl].append(m)
                    break
