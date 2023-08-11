# third party
import numpy as np
from scipy.sparse.linalg import lsmr
from scipy.special import logsumexp

# synthcity relative
from .clique_vector import CliqueVector
from .dataset import Dataset
from .factor import Factor

""" This file is experimental.
It is an attempt to re-implement and generalize the technique used in PMW^{Pub}.
https://arxiv.org/pdf/2102.08598.pdf

Notable differences:
- Shares the same interface as Private-PGM (see FactoredInference)
- Supports unbounded differential privacy, with automatic estimate of total
- Supports arbitrary measurements over the data marginals
- Solves an L2 minimization problem (by default), but can pass other loss functions if desired.
"""


def entropic_mirror_descent(loss_and_grad, x0, total, iters=250):
    logP = np.log(x0 + np.nextafter(0, 1)) + np.log(total) - np.log(x0.sum())
    P = np.exp(logP)
    P = x0 * total / x0.sum()
    loss, dL = loss_and_grad(P)
    alpha = 1.0
    begun = False

    for _ in range(iters):
        logQ = logP - alpha * dL
        logQ += np.log(total) - logsumexp(logQ)
        Q = np.exp(logQ)
        # Q = P * np.exp(-alpha*dL)
        # Q *= total / Q.sum()
        new_loss, new_dL = loss_and_grad(Q)

        if loss - new_loss >= 0.5 * alpha * dL.dot(P - Q):
            logP = logQ
            loss, dL = new_loss, new_dL
            # increase step size if we haven't already decreased it at least once
            if not begun:
                alpha *= 2
        else:
            alpha *= 0.5
            begun = True

    return np.exp(logP)


def estimate_total(measurements):
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
        return 1
    else:
        variance = 1.0 / np.sum(1.0 / variances)
        estimate = variance * np.sum(estimates / variances)
        return max(1, estimate)


class PublicInference:
    def __init__(self, public_data, metric="L2"):
        self.public_data = public_data
        self.metric = metric
        self.weights = np.ones(self.public_data.records)

    def estimate(self, measurements, total=None):
        if total is None:
            total = estimate_total(measurements)
        self.measurements = measurements
        cliques = [M[-1] for M in measurements]

        def loss_and_grad(weights):
            est = Dataset(self.public_data.df, self.public_data.domain, weights)
            mu = CliqueVector.from_data(est, cliques)
            loss, dL = self._marginal_loss(mu)
            dweights = np.zeros(weights.size)
            for cl in dL:
                idx = est.project(cl).df.values
                dweights += dL[cl].values[tuple(idx.T)]
            return loss, dweights

        # bounds = [(0,None) for _ in self.weights]
        # res = minimize(loss_and_grad, x0=self.weights, method='L-BFGS-B', jac=True, bounds=bounds)
        # self.weights = res.x

        self.weights = entropic_mirror_descent(loss_and_grad, self.weights, total)
        return Dataset(self.public_data.df, self.public_data.domain, self.weights)

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
        gradient = {cl: Factor.zeros(marginals[cl].domain) for cl in marginals}

        for Q, y, noise, cl in self.measurements:
            mu = marginals[cl]
            c = 1.0 / noise
            x = mu.datavector()
            diff = c * (Q @ x - y)
            if metric == "L1":
                loss += abs(diff).sum()
                sign = diff.sign() if hasattr(diff, "sign") else np.sign(diff)
                grad = c * (Q.T @ sign)
            else:
                loss += 0.5 * (diff @ diff)
                grad = c * (Q.T @ diff)
            gradient[cl] += Factor(mu.domain, grad)
        return float(loss), CliqueVector(gradient)
