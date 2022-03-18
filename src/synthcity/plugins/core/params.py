# stdlib
from abc import ABCMeta, abstractmethod
from typing import Any, List

# third party
import numpy as np
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.constraints import Constraints


class Params(metaclass=ABCMeta):
    @validate_arguments
    def __init__(self, name: str, low: Any, high: Any) -> None:
        self.name = name
        self.low = low
        self.high = high

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


class Categorical(Params):
    def __init__(self, name: str, choices: List[Any]) -> None:
        super().__init__(name, min(choices), max(choices))
        self.name = name
        self.choices = sorted(choices)

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
        return Constraints([(self.name, "in", list(self.choices))])


class Float(Params):
    def __init__(self, name: str, low: float, high: float) -> None:
        low = float(low)
        high = float(high)

        super().__init__(name, low, high)
        self.name = name

    def get(self) -> List[Any]:
        return [self.name, self.low, self.high]

    def sample(self) -> Any:
        return np.random.uniform(self.low, self.high)

    def has(self, val: Any) -> bool:
        return self.low <= val and val <= self.high

    def includes(self, other: "Float") -> bool:
        return self.low <= other.low and other.high <= self.high

    def as_constraint(self) -> Constraints:
        return Constraints(
            [
                (self.name, "le", self.high),
                (self.name, "ge", self.low),
            ]
        )


class Integer(Params):
    def __init__(self, name: str, low: int, high: int, step: int = 1) -> None:
        self.low = low
        self.high = high
        self.step = step

        super().__init__(name, low, high)
        self.name = name
        self.step = step
        self.choices = [val for val in range(low, high + 1, step)]

    def get(self) -> List[Any]:
        return [self.name, self.low, self.high, self.step]

    def sample(self) -> Any:
        return np.random.choice(self.choices, 1)[0]

    def has(self, val: Any) -> bool:
        return self.low <= val and val <= self.high

    def includes(self, other: "Integer") -> bool:
        return self.low <= other.low and other.high <= self.high

    def as_constraint(self) -> Constraints:
        return Constraints(
            [
                (self.name, "le", self.high),
                (self.name, "ge", self.low),
            ]
        )
