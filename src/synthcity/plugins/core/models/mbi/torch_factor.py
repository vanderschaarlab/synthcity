# third party
import numpy as np
import torch


class Factor:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, domain, values):
        """Initialize a factor over the given domain

        :param domain: the domain of the factor
        :param values: the ndarray or tensor of factor values (for each element of the domain)

        Note: values may be a flattened 1d array or a ndarray with same shape as domain
        """
        if type(values) == np.ndarray:
            values = torch.tensor(values, dtype=torch.float32, device=Factor.device)
        if domain.size() != values.nelement():
            raise ValueError("domain size does not match values size")
        if len(values.shape) != 1 and values.shape != domain.shape:
            raise ValueError("invalid shape for values array")
        self.domain = domain
        self.values = values.reshape(domain.shape).to(Factor.device)

    @staticmethod
    def zeros(domain):
        return Factor(domain, torch.zeros(domain.shape, device=Factor.device))

    @staticmethod
    def ones(domain):
        return Factor(domain, torch.ones(domain.shape, device=Factor.device))

    @staticmethod
    def random(domain):
        return Factor(domain, torch.rand(domain.shape, device=Factor.device))

    @staticmethod
    def uniform(domain):
        return Factor.ones(domain) / domain.size()

    @staticmethod
    def active(domain, structural_zeros):
        """create a factor that is 0 everywhere except in positions present in
            'structural_zeros', where it is -infinity

        :param: domain: the domain of this factor
        :param: structural_zeros: a list of values that are not possible
        """
        idx = tuple(np.array(structural_zeros).T)
        vals = torch.zeros(domain.shape, device=Factor.device)
        vals[idx] = -np.inf
        return Factor(domain, vals)

    def expand(self, domain):
        if not domain.contains(self.domain):
            raise AssertionError("expanded domain must contain current domain")
        dims = len(domain) - len(self.domain)
        values = self.values.view(self.values.size() + tuple([1] * dims))
        ax = domain.axes(self.domain.attrs)
        # need to find replacement for moveaxis
        ax = ax + tuple(i for i in range(len(domain)) if i not in ax)
        ax = tuple(np.argsort(ax))
        values = values.permute(ax)
        values = values.expand(domain.shape)
        return Factor(domain, values)

    def transpose(self, attrs):
        if set(attrs) != set(self.domain.attrs):
            raise AssertionError("attrs must be same as domain attributes")
        newdom = self.domain.project(attrs)
        ax = newdom.axes(self.domain.attrs)
        ax = tuple(np.argsort(ax))
        values = self.values.permute(ax)
        return Factor(newdom, values)

    def project(self, attrs, agg="sum"):
        """
        project the factor onto a list of attributes (in order)
        using either sum or logsumexp to aggregate along other attributes
        """
        if agg not in ["sum", "logsumexp"]:
            raise ValueError("agg must be sum or logsumexp")
        marginalized = self.domain.marginalize(attrs)
        if agg == "sum":
            ans = self.sum(marginalized.attrs)
        elif agg == "logsumexp":
            ans = self.logsumexp(marginalized.attrs)
        return ans.transpose(attrs)

    def sum(self, attrs=None):
        if attrs is None:
            return float(self.values.sum())
        elif attrs == tuple():
            return self
        axes = self.domain.axes(attrs)
        values = self.values.sum(dim=axes)
        newdom = self.domain.marginalize(attrs)
        return Factor(newdom, values)

    def logsumexp(self, attrs=None):
        if attrs is None:
            return float(
                self.values.logsumexp(dim=tuple(range(len(self.values.shape))))
            )
        elif attrs == tuple():
            return self
        axes = self.domain.axes(attrs)
        values = self.values.logsumexp(dim=axes)
        newdom = self.domain.marginalize(attrs)
        return Factor(newdom, values)

    def logaddexp(self, other):
        return NotImplementedError

    def max(self, attrs=None):
        if attrs is None:
            return float(self.values.max())
        return NotImplementedError  # torch.max does not behave like numpy

    def condition(self, evidence):
        """evidence is a dictionary where
        keys are attributes, and
        values are elements of the domain for that attribute"""
        slices = [evidence[a] if a in evidence else slice(None) for a in self.domain]
        newdom = self.domain.marginalize(evidence.keys())
        values = self.values[tuple(slices)]
        return Factor(newdom, values)

    def copy(self, out=None):
        if out is None:
            return Factor(self.domain, self.values.clone())
        np.copyto(out.values, self.values)
        return out

    def __mul__(self, other):
        if np.isscalar(other):
            return Factor(self.domain, other * self.values)
        newdom = self.domain.merge(other.domain)
        factor1 = self.expand(newdom)
        factor2 = other.expand(newdom)
        return Factor(newdom, factor1.values * factor2.values)

    def __add__(self, other):
        if np.isscalar(other):
            return Factor(self.domain, other + self.values)
        newdom = self.domain.merge(other.domain)
        factor1 = self.expand(newdom)
        factor2 = other.expand(newdom)
        return Factor(newdom, factor1.values + factor2.values)

    def __iadd__(self, other):
        if np.isscalar(other):
            self.values += other
            return self
        factor2 = other.expand(self.domain)
        self.values += factor2.values
        return self

    def __imul__(self, other):
        if np.isscalar(other):
            self.values *= other
            return self
        factor2 = other.expand(self.domain)
        self.values *= factor2.values
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if np.isscalar(other):
            return Factor(self.domain, self.values - other)
        zero = torch.tensor(0.0, device=Factor.device)
        inf = torch.tensor(np.inf, device=Factor.device)
        values = torch.where(other.values == -inf, zero, -other.values)
        other = Factor(other.domain, values)
        return self + other

    def __truediv__(self, other):
        if np.isscalar(other):
            return self * (1.0 / other)
        tmp = other.expand(self.domain)
        vals = torch.div(self.values, tmp.values)
        vals[tmp.values <= 0] = 0.0
        return Factor(self.domain, vals)

    def exp(self, out=None):
        if out is None:
            return Factor(self.domain, self.values.exp())
        torch.exp(self.values, out=out.values)
        return out

    def log(self, out=None):
        if out is None:
            return Factor(self.domain, torch.log(self.values + 1e-100))
        torch.log(self.values, out=out.values)
        return out

    def datavector(self, flatten=True):
        """Materialize the data vector as a numpy array"""
        ans = self.values.to("cpu").numpy()
        return ans.flatten() if flatten else ans
