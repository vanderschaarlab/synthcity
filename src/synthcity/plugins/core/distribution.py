# stdlib
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional

# third party
import numpy as np
import pandas as pd
from pydantic import BaseModel, validator

# synthcity absolute
from synthcity.plugins.core.constraints import Constraints


class Distribution(BaseModel, metaclass=ABCMeta):
    data: Optional[pd.Series] = None
    epsilon: float = 1.0
    use_dp: bool = False
    marginal_distribution: Optional[pd.Series] = None
    dp_marginal_distribution: Optional[pd.Series] = None
    name: str

    class Config:
        arbitrary_types_allowed = True

    @validator("marginal_distribution", always=True)
    def _validate_marginal_distribution(cls: Any, v: Any, values: Dict) -> Dict:
        if "data" not in values or values["data"] is None:
            return v

        data = values["data"]
        if not isinstance(data, pd.Series):
            raise ValueError(f"Invalid data type {type(data)}")

        marginal = data.value_counts(dropna=False)
        del values["data"]
        return marginal

    def marginal_states(self) -> Optional[List]:
        if self.marginal_distribution is None:
            return None

        return self.marginal_distribution.index.values

    def marginal_probabilities(self) -> Optional[List]:
        if self.marginal_distribution is None:
            return None

        return (
            self.marginal_distribution.values / self.marginal_distribution.values.sum()
        )

    def sample_marginal(self, count: int = 1) -> Any:
        if self.marginal_distribution is None:
            return None

        return np.random.choice(
            self.marginal_states(),
            count,
            p=self.marginal_probabilities(),
        )

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

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        ...


class CategoricalDistribution(Distribution):
    choices: list = []

    @validator("choices", always=True)
    def _validate_choices(cls: Any, v: List, values: Dict) -> List:
        mkey = "marginal_distribution"
        if mkey in values and values[mkey] is not None:
            return list(values[mkey].index)

        if len(v) == 0:
            raise ValueError(
                "Invalid choices for CategoricalDistribution. Provide data or choices params"
            )
        return v

    def get(self) -> List[Any]:
        return [self.name, self.choices]

    def sample(self, count: int = 1) -> Any:
        msamples = self.sample_marginal(count)
        if msamples is not None:
            return msamples

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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CategoricalDistribution):
            return False

        return self.name == other.name and set(self.choices) == set(other.choices)


class FloatDistribution(Distribution):
    low: float = np.iinfo(np.int32).min
    high: float = np.iinfo(np.int32).max

    @validator("low", always=True)
    def _validate_low_thresh(cls: Any, v: float, values: Dict) -> float:
        mkey = "marginal_distribution"
        if mkey in values and values[mkey] is not None:
            return values[mkey].index.min()

        return v

    @validator("high", always=True)
    def _validate_high_thresh(cls: Any, v: float, values: Dict) -> float:
        mkey = "marginal_distribution"
        if mkey in values and values[mkey] is not None:
            return values[mkey].index.max()

        return v

    def get(self) -> List[Any]:
        return [self.name, self.low, self.high]

    def sample(self, count: int = 1) -> Any:
        msamples = self.sample_marginal(count)
        if msamples is not None:
            return msamples
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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FloatDistribution):
            return False

        return (
            self.name == other.name
            and self.low == other.low
            and self.high == other.high
        )


class IntegerDistribution(Distribution):
    low: int = np.iinfo(np.int32).min
    high: int = np.iinfo(np.int32).max
    step: int = 1

    @validator("low", always=True)
    def _validate_low_thresh(cls: Any, v: int, values: Dict) -> int:
        mkey = "marginal_distribution"
        if mkey in values and values[mkey] is not None:
            return int(values[mkey].index.min())

        return v

    @validator("high", always=True)
    def _validate_high_thresh(cls: Any, v: int, values: Dict) -> int:
        mkey = "marginal_distribution"
        if mkey in values and values[mkey] is not None:
            return int(values[mkey].index.max())
        return v

    def get(self) -> List[Any]:
        return [self.name, self.low, self.high, self.step]

    def sample(self, count: int = 1) -> Any:
        msamples = self.sample_marginal(count)
        if msamples is not None:
            return msamples

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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, IntegerDistribution):
            return False

        return (
            self.name == other.name
            and self.low == other.low
            and self.high == other.high
        )


def constraint_to_distribution(constraints: Constraints, feature: str) -> Distribution:
    dist_name, dist_args = constraints.feature_params(feature)

    if dist_name == "categorical":
        dist_template = CategoricalDistribution
    elif dist_name == "integer":
        dist_template = IntegerDistribution
    else:
        dist_template = FloatDistribution

    return dist_template(name=feature, **dist_args)
