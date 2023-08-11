# stdlib
import itertools
import math
import platform
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

if platform.python_version() < "3.9":
    # stdlib
    from typing import Iterable
else:
    from collections.abc import Iterable

# third party
import numpy as np
from scipy.special import softmax

# synthcity absolute
import synthcity.logger as log

# synthcity relative
from .mbi.dataset import Dataset
from .mbi.domain import Domain
from .mbi.graphical_model import GraphicalModel
from .mbi.identity import Identity
from .mbi.inference import FactoredInference


def powerset(iterable: Iterable) -> Iterable:
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)
    )


def downward_closure(Ws: List[Tuple]) -> List:
    ans: Set = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))


def hypothetical_model_size(domain: Domain, cliques: List[Union[Tuple, List]]) -> float:
    model = GraphicalModel(domain, cliques)
    return model.size * 8 / 2**20


def compile_workload(workload: List[Tuple]) -> Dict:
    def score(cl: Tuple) -> int:
        return sum(len(set(cl) & set(ax)) for ax in workload)

    return {cl: score(cl) for cl in downward_closure(workload)}


def filter_candidates(
    candidates: Dict, model: GraphicalModel, size_limit: Union[float, int]
) -> Dict:
    ans = {}
    free_cliques = downward_closure(model.cliques)
    for cl in candidates:
        cond1 = (
            hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        )
        cond2 = cl in free_cliques
        if cond1 or cond2:
            ans[cl] = candidates[cl]
    return ans


def cdp_delta(rho: Union[float, int], eps: Union[float, int]) -> Union[float, int]:
    if rho < 0:
        raise ValueError("rho must be positive")
    if eps < 0:
        raise ValueError("eps must be positive")
    if rho == 0:
        return 0  # degenerate case

    # search for best alpha
    # Note that any alpha in (1,infty) yields a valid upper bound on delta
    # Thus if this search is slightly "incorrect" it will only result in larger delta (still valid)
    # This code has two "hacks".
    # First the binary search is run for a pre-specificed length.
    # 1000 iterations should be sufficient to converge to a good solution.
    # Second we set a minimum value of alpha to avoid numerical stability issues.
    # Note that the optimal alpha is at least (1+eps/rho)/2. Thus we only hit this constraint
    # when eps<=rho or close to it. This is not an interesting parameter regime, as you will
    # inherently get large delta in this regime.
    amin = 1.01  # don't let alpha be too small, due to numerical stability
    amax = (eps + 1) / (2 * rho) + 2
    for i in range(1000):  # should be enough iterations
        alpha = (amin + amax) / 2
        derivative = (2 * alpha - 1) * rho - eps + math.log1p(-1.0 / alpha)
        if derivative < 0:
            amin = alpha
        else:
            amax = alpha
    # now calculate delta
    delta = math.exp(
        (alpha - 1) * (alpha * rho - eps) + alpha * math.log1p(-1 / alpha)
    ) / (alpha - 1.0)
    return min(delta, 1.0)  # delta<=1 always


def cdp_rho(eps: float, delta: float) -> float:
    if eps < 0:
        raise ValueError("eps must be positive")
    if delta <= 0:
        raise ValueError("delta must be positive")

    if delta >= 1:
        return 0.0  # if delta>=1 anything goes
    rho_min = 0.0  # maintain cdp_delta(rho,eps)<=delta
    rho_max = eps + 1  # maintain cdp_delta(rho_max,eps)>delta
    for i in range(1000):
        rho = (rho_min + rho_max) / 2
        if cdp_delta(rho, eps) <= delta:
            rho_min = rho
        else:
            rho_max = rho
    return rho_min


class Mechanism:
    def __init__(self, epsilon: float, delta: float):
        """
        Base class for a mechanism.
        :param epsilon: privacy parameter
        :param delta: privacy parameter
        :param prng: pseudo random number generator
        """
        self.epsilon = epsilon
        self.delta = delta
        self.rho = 0 if delta == 0 else cdp_rho(epsilon, delta)
        self.prng = np.random

    def run(self, dataset: Dataset, workload: List[Tuple]) -> Any:
        pass

    # def generalized_exponential_mechanism(
    #     self, qualities, sensitivities, epsilon, t=None, base_measure=None
    # ):
    #     def generalized_em_scores(q, ds, t):
    #         def pareto_efficient(costs: np.ndarray) -> int:
    #             eff = np.ones(costs.shape[0], dtype=bool)
    #             for i, c in enumerate(costs):
    #                 if eff[i]:
    #                     eff[eff] = np.any(
    #                         costs[eff] <= c, axis=1
    #                     )  # Keep any point with a lower cost
    #             return np.nonzero(eff)[0]

    #         q = -q
    #         idx = pareto_efficient(np.vstack([q, ds]).T)
    #         r = q + t * ds
    #         r = r[:, None] - r[idx][None, :]
    #         z = ds[:, None] + ds[idx][None, :]
    #         s = (r / z).max(axis=1)
    #         return -s

    #     if t is None:
    #         t = 2 * np.log(len(qualities) / 0.5) / epsilon
    #     if isinstance(qualities, dict):
    #         keys = list(qualities.keys())
    #         qualities = np.array([qualities[key] for key in keys])
    #         sensitivities = np.array([sensitivities[key] for key in keys])
    #         if base_measure is not None:
    #             base_measure = np.log([base_measure[key] for key in keys])
    #     else:
    #         keys = np.arange(qualities.size)
    #     scores = generalized_em_scores(qualities, sensitivities, t)
    #     key = self.exponential_mechanism(
    #         scores, epsilon, 1.0, base_measure=base_measure
    #     )
    #     return keys[key]

    # def permute_and_flip(self, qualities, epsilon, sensitivity=1.0):
    #     """Sample a candidate from the permute-and-flip mechanism"""
    #     q = qualities - qualities.max()
    #     p = np.exp(0.5 * epsilon / sensitivity * q)
    #     for i in np.random.permutation(p.size):
    #         if np.random.rand() <= p[i]:
    #             return i

    def exponential_mechanism(
        self,
        qualities: Union[Dict, np.ndarray, Any],
        epsilon: float,
        sensitivity: Union[float, int] = 1.0,
        base_measure: Optional[Dict] = None,
    ) -> np.ndarray:
        if isinstance(qualities, dict):
            keys = list(qualities.keys())
            qualities = cast(np.ndarray, np.array([qualities[key] for key in keys]))
            if base_measure is not None:
                base_measure = np.log([base_measure[key] for key in keys])
        else:
            qualities = cast(np.ndarray, np.array(qualities))
            keys = np.arange(qualities.size)

        """ Sample a candidate from the permute-and-flip mechanism """
        q = qualities - qualities.max()
        if base_measure is None:
            p = softmax(0.5 * epsilon / sensitivity * q)
        else:
            p = softmax(0.5 * epsilon / sensitivity * q + base_measure)

        return keys[self.prng.choice(p.size, p=p)]

    # def gaussian_noise_scale(self, l2_sensitivity, epsilon, delta):
    #     """Return the Gaussian noise necessary to attain (epsilon, delta)-DP"""
    #     if self.bounded:
    #         l2_sensitivity *= 2.0
    #     return (
    #         l2_sensitivity
    #         * privacy_calibrator.ana_gaussian_mech(epsilon, delta)["sigma"]
    #     )

    # def laplace_noise_scale(self, l1_sensitivity, epsilon):
    #     """Return the Laplace noise necessary to attain epsilon-DP"""
    #     if self.bounded:
    #         l1_sensitivity *= 2.0
    #     return l1_sensitivity / epsilon

    def gaussian_noise(self, sigma: float, size: Union[int, Tuple]) -> np.ndarray:
        """Generate iid Gaussian noise  of a given scale and size"""
        return self.prng.normal(0, sigma, size)

    # def laplace_noise(self, b, size):
    #     """Generate iid Laplace noise  of a given scale and size"""
    #     return self.prng.laplace(0, b, size)

    # def best_noise_distribution(self, l1_sensitivity, l2_sensitivity, epsilon, delta):
    #     """Adaptively determine if Laplace or Gaussian noise will be better, and
    #     return a function that samples from the appropriate distribution"""
    #     b = self.laplace_noise_scale(l1_sensitivity, epsilon)
    #     sigma = self.gaussian_noise_scale(l2_sensitivity, epsilon, delta)
    #     if np.sqrt(2) * b < sigma:
    #         return partial(self.laplace_noise, b)
    #     return partial(self.gaussian_noise, sigma)


class AIM(Mechanism):
    def __init__(
        self,
        epsilon: float,
        delta: float,
        rounds: Optional[Union[int, float]] = None,
        max_model_size: int = 80,
        structural_zeros: Dict = {},
    ):
        super(AIM, self).__init__(epsilon, delta)
        self.rounds = rounds
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros

    def worst_approximated(
        self,
        candidates: Dict,
        answers: Dict,
        model: GraphicalModel,
        eps: float,
        sigma: float,
    ) -> np.ndarray:
        errors = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl)
            xest = model.project(cl).datavector()
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt)
        max_sensitivity = max(
            sensitivity.values()
        )  # if all weights are 0, could be a problem
        return self.exponential_mechanism(errors, eps, max_sensitivity)

    def run(self, data: Dataset, W: List) -> Dataset:
        rounds = self.rounds or 16 * len(data.domain)
        workload = [cl for cl, _ in W]
        candidates = compile_workload(workload)
        answers = {cl: data.project(cl).datavector() for cl in candidates}

        oneway = [cl for cl in candidates if len(cl) == 1]

        sigma = np.sqrt(rounds / (2 * 0.9 * self.rho))
        epsilon = np.sqrt(8 * 0.1 * self.rho / rounds)

        measurements = []
        log.info("Initial Sigma", sigma)
        rho_used = len(oneway) * 0.5 / sigma**2
        for cl in oneway:
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, x.size)
            identity_I = Identity(y.size)
            measurements.append((identity_I, y, sigma, cl))

        # backend = "torch" if torch.cuda.is_available() else "cpu" # TODO: fix torch backend option
        zeros = self.structural_zeros
        engine = FactoredInference(
            data.domain,
            # backend=backend,
            iters=1000,
            warm_start=True,
            structural_zeros=zeros,
        )
        model = engine.estimate(measurements)

        t = 0
        terminate = False
        while not terminate:
            t += 1
            if self.rho - rho_used < 2 * (0.5 / sigma**2 + 1.0 / 8 * epsilon**2):
                # Just use up whatever remaining budget there is for one last round
                remaining = self.rho - rho_used
                sigma = np.sqrt(1 / (2 * 0.9 * remaining))
                epsilon = np.sqrt(8 * 0.1 * remaining)
                terminate = True

            rho_used += 1.0 / 8 * epsilon**2 + 0.5 / sigma**2
            size_limit = self.max_model_size * rho_used / self.rho
            small_candidates = filter_candidates(candidates, model, size_limit)
            cl = self.worst_approximated(
                small_candidates, answers, model, epsilon, sigma
            )

            n = data.domain.size(cl)
            Q = Identity(n)
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, n)
            measurements.append((Q, y, sigma, cl))
            z = model.project(cl).datavector()

            model = engine.estimate(measurements)
            w = model.project(cl).datavector()
            log.info("Selected", cl, "Size", n, "Budget Used", rho_used / self.rho)
            if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
                log.warning(f"Reducing sigma: {sigma/2}")
                sigma /= 2
                epsilon *= 2

        log.info("Generating Data...")
        engine.iters = 2500
        model = engine.estimate(measurements)
        synth = model.synthetic_data()

        return synth


def default_params() -> Dict[str, Any]:
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params: Dict[str, Any] = {}
    params["dataset"] = "../data/adult.csv"  # TODO: Generalize
    params["domain"] = "../data/adult-domain.json"  # TODO: Generalize
    params["epsilon"] = 1.0
    params["delta"] = 1e-9
    params["noise"] = "laplace"
    params["max_model_size"] = 80
    params["degree"] = 2
    params["num_marginals"] = None
    params["max_cells"] = 10000

    return params
