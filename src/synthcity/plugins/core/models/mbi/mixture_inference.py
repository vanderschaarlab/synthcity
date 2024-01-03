# third party
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import vjp
from jax.nn import softmax as jax_softmax
from mbi import Dataset
from scipy.sparse.linalg import lsmr

""" This file is experimental.

It is a close approximation to the method described in RAP (https://arxiv.org/abs/2103.06641)
and an even closer approximation to RAP^{softmax} (https://arxiv.org/abs/2106.07153)

Notable differences:
- Code now shares the same interface as Private-PGM (see FactoredInference)
- Named model "MixtureOfProducts", as that is one interpretation for the relaxed tabular format
(at least when softmax is used).
- Added support for unbounded-DP, with automatic estimate of total.
"""


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


def adam(loss_and_grad, x0, iters=250):
    a = 1.0
    b1, b2 = 0.9, 0.999
    eps = 10e-8

    x = x0
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for t in range(1, iters + 1):
        l, g = loss_and_grad(x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        x = x - a * mhat / (np.sqrt(vhat) + eps)
    return x


def synthetic_col(counts, total):
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


class MixtureOfProducts:
    def __init__(self, products, domain, total):
        self.products = products
        self.domain = domain
        self.total = total
        self.num_components = next(iter(products.values())).shape[0]

    def project(self, cols):
        products = {col: self.products[col] for col in cols}
        domain = self.domain.project(cols)
        return MixtureOfProducts(products, domain, self.total)

    def datavector(self, flatten=True):
        letters = "bcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[
            : len(self.domain)
        ]
        formula = (
            ",".join(["a%s" % letter for letter in letters]) + "->" + "".join(letters)
        )
        components = [self.products[col] for col in self.domain]
        ans = np.einsum(formula, *components) * self.total / self.num_components
        return ans.flatten() if flatten else ans

    def synthetic_data(self, rows=None):
        total = rows or int(self.total)
        subtotal = total // self.num_components + 1

        dfs = []
        for i in range(self.num_components):
            df = pd.DataFrame()
            for col in self.products:
                counts = self.products[col][i]
                df[col] = synthetic_col(counts, subtotal)
            dfs.append(df)

        df = pd.concat(dfs).sample(frac=1).reset_index(drop=True)[:total]
        return Dataset(df, self.domain)


class MixtureInference:
    def __init__(
        self, domain, components=10, metric="L2", iters=2500, warm_start=False
    ):
        """
        :param domain: A Domain object
        :param components: The number of mixture components
        :metric: The metric to use for the loss function (can be callable)
        """
        self.domain = domain
        self.components = components
        self.metric = metric
        self.iters = iters
        self.warm_start = warm_start
        self.params = np.random.normal(
            loc=0, scale=0.25, size=sum(domain.shape) * components
        )

    def estimate(self, measurements, total=None, alpha=0.1):
        if total is None:
            total = estimate_total(measurements)
        self.measurements = measurements
        cliques = [M[-1] for M in measurements]
        letters = "bcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

        def get_products(params):
            products = {}
            idx = 0
            for col in self.domain:
                n = self.domain[col]
                k = self.components
                products[col] = jax_softmax(
                    params[idx : idx + k * n].reshape(k, n), axis=1
                )
                idx += k * n
            return products

        def marginals_from_params(params):
            products = get_products(params)
            mu = {}
            for cl in cliques:
                let = letters[: len(cl)]
                formula = (
                    ",".join(["a%s" % letter for letter in let]) + "->" + "".join(let)
                )
                components = [products[col] for col in cl]
                ans = jnp.einsum(formula, *components) * total / self.components
                mu[cl] = ans.flatten()
            return mu

        def loss_and_grad(params):
            # For computing dL / dmu we will use ordinary numpy so as to support scipy sparse and linear operator inputs
            # For computing dL / dparams we will use jax to avoid manually deriving gradients
            params = jnp.array(params)
            mu, backprop = vjp(marginals_from_params, params)
            mu = {cl: np.array(mu[cl]) for cl in cliques}
            loss, dL = self._marginal_loss(mu)
            dL = {cl: jnp.array(dL[cl]) for cl in cliques}
            dparams = backprop(dL)
            return loss, np.array(dparams[0])

        if not self.warm_start:
            self.params = np.random.normal(
                loc=0, scale=0.25, size=sum(self.domain.shape) * self.components
            )
        self.params = adam(loss_and_grad, self.params, iters=self.iters)
        products = get_products(self.params)
        return MixtureOfProducts(products, self.domain, total)

    def _marginal_loss(self, marginals, metric=None):
        """Compute the loss and gradient for a given dictionary of marginals

        :param marginals: A dictionary with keys as projections and values as Factors
        :return loss: the loss value
        :return grad: A dictionary with gradient for each marginal
        """
        if metric is None:
            metric = self.metric

        loss = 0.0
        gradient = {cl: np.zeros_like(marginals[cl]) for cl in marginals}

        for Q, y, noise, cl in self.measurements:
            x = marginals[cl]
            c = 1.0 / noise
            diff = c * (Q @ x - y)
            if metric == "L1":
                loss += abs(diff).sum()
                sign = diff.sign() if hasattr(diff, "sign") else np.sign(diff)
                grad = c * (Q.T @ sign)
            else:
                loss += 0.5 * (diff @ diff)
                grad = c * (Q.T @ diff)
            gradient[cl] += grad

        return float(loss), gradient
