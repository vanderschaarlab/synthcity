# stdlib
from abc import ABCMeta, abstractmethod
from typing import Any, List

# third party
import numpy as np
from pydantic import BaseModel

# synthcity absolute
from synthcity.plugins.core.constraints import Constraints


class Distribution(BaseModel, metaclass=ABCMeta):
    name: str

    @abstractmethod
    def get(self) -> List[Any]:
        """Return the metadata of the Distribution."""
        ...

    @abstractmethod
    def sample(self, count: int = 1) -> Any:
        """Sample a value from the Distribution."""
        ...

    @abstractmethod
    def includes(self, other: "Distribution") -> bool:
        """Test if another Distribution is included in the local one."""
        ...

    @abstractmethod
    def has(self, val: Any) -> bool:
        """Test if a value is included in the Distribution."""
        ...

    @abstractmethod
    def as_constraint(self) -> Constraints:
        """Convert the Distribution to a set of Constraints."""
        ...

    @abstractmethod
    def min(self) -> Any:
        "Get the min value of the distribution"
        ...

    @abstractmethod
    def max(self) -> Any:
        "Get the max value of the distribution"
        ...


class CategoricalDistribution(Distribution):
    choices: list

    def get(self) -> List[Any]:
        return [self.name, self.choices]

    def sample(self, count: int = 1) -> Any:
        return np.random.choice(self.choices, count)

    def has(self, val: Any) -> bool:
        return val in self.choices

    def includes(self, other: "Distribution") -> bool:
        if not isinstance(other, CategoricalDistribution):
            return False
        return set(other.choices).issubset(set(self.choices))

    def as_constraint(self) -> Constraints:
        return Constraints(rules=[(self.name, "in", list(self.choices))])

    def min(self) -> Any:
        return min(self.choices)

    def max(self) -> Any:
        return max(self.choices)


class FloatDistribution(Distribution):
    low: float = -np.inf
    high: float = np.inf

    def get(self) -> List[Any]:
        return [self.name, self.low, self.high]

    def sample(self, count: int = 1) -> Any:
        return np.random.uniform(self.low, self.high, count)

    def has(self, val: Any) -> bool:
        return self.low <= val and val <= self.high

    def includes(self, other: "Distribution") -> bool:
        return self.min() <= other.min() and other.max() <= self.max()

    def as_constraint(self) -> Constraints:
        return Constraints(
            rules=[
                (self.name, "le", self.high),
                (self.name, "ge", self.low),
                (self.name, "dtype", "float"),
            ]
        )

    def min(self) -> Any:
        return self.low

    def max(self) -> Any:
        return self.high


class IntegerDistribution(Distribution):
    low: int = -np.inf
    high: int = np.inf
    step: int = 1

    def get(self) -> List[Any]:
        return [self.name, self.low, self.high, self.step]

    def sample(self, count: int = 1) -> Any:
        choices = [val for val in range(self.low, self.high + 1, self.step)]
        return np.random.choice(choices, count)

    def has(self, val: Any) -> bool:
        return self.low <= val and val <= self.high

    def includes(self, other: "Distribution") -> bool:
        return self.min() <= other.min() and other.max() <= self.max()

    def as_constraint(self) -> Constraints:
        return Constraints(
            rules=[
                (self.name, "le", self.high),
                (self.name, "ge", self.low),
                (self.name, "dtype", "int"),
            ]
        )

    def min(self) -> Any:
        return self.low

    def max(self) -> Any:
        return self.high


def constraint_to_distribution(constraints: Constraints, feature: str) -> Distribution:
    rules = constraints.feature_constraints(feature)

    dist_template = FloatDistribution
    dist_args = {"low": -np.inf, "high": np.inf}

    for op, value in rules:
        if op == "in":
            dist_template = CategoricalDistribution
            dist_args = {"choices": value}
            break
        elif op == "dtype" and value == "int":
            dist_template = IntegerDistribution
        elif op == "le" and value < dist_args["high"]:
            dist_args["high"] = value
        elif op == "lt" and value < dist_args["high"]:
            dist_args["high"] = value - 1
        elif op == "ge" and dist_args["low"] < value:
            dist_args["low"] = value
        elif op == "gt" and dist_args["low"] < value:
            dist_args["low"] = value + 1
        elif op == "eq":
            dist_args["low"] = value
            dist_args["high"] = value

    return dist_template(name=feature, **dist_args)
