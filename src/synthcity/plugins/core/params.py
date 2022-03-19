# stdlib
from abc import ABCMeta, abstractmethod
from typing import Any, List

# third party
import numpy as np
from pydantic import BaseModel

# synthcity absolute
from synthcity.plugins.core.constraints import Constraints


class Params(metaclass=ABCMeta):
    @abstractmethod
    def get(self) -> List[Any]:
        ...

    @abstractmethod
    def sample(self) -> Any:
        ...

    @abstractmethod
    def includes(self, other: "Params") -> bool:
        ...

    @abstractmethod
    def has(self, val: Any) -> bool:
        ...

    @abstractmethod
    def as_constraint(self) -> Constraints:
        ...

    @abstractmethod
    def min(self) -> Any:
        ...

    @abstractmethod
    def max(self) -> Any:
        ...


class Categorical(BaseModel, Params):
    name: str
    choices: list

    def get(self) -> List[Any]:
        return [self.name, self.choices]

    def sample(self) -> Any:
        return np.random.choice(self.choices, 1)[0]

    def has(self, val: Any) -> bool:
        return val in self.choices

    def includes(self, other: "Categorical") -> bool:
        if not isinstance(other, Categorical):
            return False
        return set(other.choices).issubset(set(self.choices))

    def as_constraint(self) -> Constraints:
        return Constraints(rules=[(self.name, "in", list(self.choices))])

    def min(self) -> Any:
        return min(self.choices)

    def max(self) -> Any:
        return max(self.choices)


class Float(BaseModel, Params):
    name: str
    low: float
    high: float

    def get(self) -> List[Any]:
        return [self.name, self.low, self.high]

    def sample(self) -> Any:
        return np.random.uniform(self.low, self.high)

    def has(self, val: Any) -> bool:
        return self.low <= val and val <= self.high

    def includes(self, other: "Params") -> bool:
        return self.min() <= other.min() and other.max() <= self.max()

    def as_constraint(self) -> Constraints:
        return Constraints(
            rules=[
                (self.name, "le", self.high),
                (self.name, "ge", self.low),
            ]
        )

    def min(self) -> Any:
        return self.low

    def max(self) -> Any:
        return self.high


class Integer(BaseModel, Params):
    name: str
    low: int
    high: int
    step: int = 1

    def get(self) -> List[Any]:
        return [self.name, self.low, self.high, self.step]

    def sample(self) -> Any:
        choices = [val for val in range(self.low, self.high + 1, self.step)]
        return np.random.choice(choices, 1)[0]

    def has(self, val: Any) -> bool:
        return self.low <= val and val <= self.high

    def includes(self, other: "Params") -> bool:
        return self.min() <= other.min() and other.max() <= self.max()

    def as_constraint(self) -> Constraints:
        return Constraints(
            rules=[
                (self.name, "le", self.high),
                (self.name, "ge", self.low),
            ]
        )

    def min(self) -> Any:
        return self.low

    def max(self) -> Any:
        return self.high
