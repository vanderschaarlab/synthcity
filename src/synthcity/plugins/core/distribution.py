# stdlib
from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# third party
import numpy as np
import pandas as pd
from pydantic import BaseModel, validator

# synthcity absolute
from synthcity.plugins.core.constraints import Constraints


class Distribution(BaseModel, metaclass=ABCMeta):
    """
    .. inheritance-diagram:: synthcity.plugins.core.distribution.Distribution
        :parts: 1


    Base class of all Distributions.

    The Distribution class characterizes the **empirical** marginal distribution of the feature.
    Each derived class must implement the following methods:
        get() - Return the metadata of the Distribution.
        sample() - Sample a value from the Distribution.
        includes() - Test if another Distribution is included in the local one.
        has() - Test if a value is included in the support of the Distribution.
        as_constraint() - Convert the Distribution to a set of Constraints.
        min() - Return the minimum of the support.
        max() - Return the maximum of the support.
        __eq__() - Testing equality of two Distributions.
        dtype() - Return the data type

    Examples of derived classes include CategoricalDistribution, FloatDistribution, and IntegerDistribution.
    """

    name: str
    data: Optional[pd.Series] = None
    random_state: int = 0
    # DP parameters
    marginal_distribution: Optional[pd.Series] = None

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
        np.random.seed(self.random_state)

        if self.marginal_distribution is None:
            return None

        return np.random.choice(
            self.marginal_states(),
            count,
            p=self.marginal_probabilities(),
        ).tolist()

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

    @abstractmethod
    def dtype(self) -> str:
        ...


class CategoricalDistribution(Distribution):
    """
    .. inheritance-diagram:: synthcity.plugins.core.distribution.CategoricalDistribution
        :parts: 1
    """

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
        np.random.seed(self.random_state)
        msamples = self.sample_marginal(count)
        if msamples is not None:
            return msamples

        return np.random.choice(self.choices, count).tolist()

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

    def dtype(self) -> str:
        types = {
            "object": 0,
            "float": 0,
            "int": 0,
        }
        for v in self.choices:
            if isinstance(v, float):
                types["float"] += 1
            elif isinstance(v, int):
                types["int"] += 1
            else:
                types["object"] += 1

        for t in types:
            if types[t] != 0:
                return t

        return "object"


class FloatDistribution(Distribution):
    """
    .. inheritance-diagram:: synthcity.plugins.core.distribution.FloatDistribution
        :parts: 1
    """

    low: float = np.iinfo(np.int64).min
    high: float = np.iinfo(np.int64).max

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
        np.random.seed(self.random_state)
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

    def dtype(self) -> str:
        return "float"


class IntegerDistribution(Distribution):
    """
    .. inheritance-diagram:: synthcity.plugins.core.distribution.IntegerDistribution
        :parts: 1
    """

    low: int = np.iinfo(np.int64).min
    high: int = np.iinfo(np.int64).max
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
        np.random.seed(self.random_state)
        msamples = self.sample_marginal(count)
        if msamples is not None:
            return msamples

        choices = [val for val in range(self.low, self.high + 1, self.step)]
        return np.random.choice(choices, count).tolist()

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

    def dtype(self) -> str:
        return "int"


OFFSET = 120


class DatetimeDistribution(Distribution):
    """
    .. inheritance-diagram:: synthcity.plugins.core.distribution.DatetimeDistribution
        :parts: 1
    """

    low: datetime = datetime.utcfromtimestamp(0)
    high: datetime = datetime.now()

    @validator("low", always=True)
    def _validate_low_thresh(cls: Any, v: datetime, values: Dict) -> datetime:
        mkey = "marginal_distribution"
        if mkey in values and values[mkey] is not None:
            v = values[mkey].index.min()

        return v - timedelta(seconds=OFFSET)

    @validator("high", always=True)
    def _validate_high_thresh(cls: Any, v: datetime, values: Dict) -> datetime:
        mkey = "marginal_distribution"
        if mkey in values and values[mkey] is not None:
            v = values[mkey].index.max()

        return v + timedelta(seconds=OFFSET)

    def get(self) -> List[Any]:
        return [self.name, self.low, self.high]

    def sample(self, count: int = 1) -> Any:
        np.random.seed(self.random_state)
        msamples = self.sample_marginal(count)
        if msamples is not None:
            return msamples

        samples = np.random.uniform(
            datetime.timestamp(self.low), datetime.timestamp(self.high), count
        )

        samples_dt = []
        for s in samples:
            samples_dt.append(datetime.fromtimestamp(s))

        return samples_dt

    def has(self, val: datetime) -> bool:
        return self.low <= val and val <= self.high

    def includes(self, other: "Distribution") -> bool:
        return self.min() - timedelta(
            seconds=OFFSET
        ) <= other.min() and other.max() <= self.max() + timedelta(seconds=OFFSET)

    def as_constraint(self) -> Constraints:
        return Constraints(
            rules=[
                (self.name, "le", self.high),
                (self.name, "ge", self.low),
                (self.name, "dtype", "datetime"),
            ]
        )

    def min(self) -> Any:
        return self.low

    def max(self) -> Any:
        return self.high

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DatetimeDistribution):
            return False

        return (
            self.name == other.name
            and self.low == other.low
            and self.high == other.high
        )

    def dtype(self) -> str:
        return "datetime"


def constraint_to_distribution(constraints: Constraints, feature: str) -> Distribution:
    """Infer Distribution from Constraints.

    Args:
        constraints: Constraints
            The Constraints on features.
        feature: str
            The name of the feature in question.

    Returns:
        The inferred Distribution.
    """
    dist_name, dist_args = constraints.feature_params(feature)

    if dist_name == "categorical":
        dist_template = CategoricalDistribution
    elif dist_name == "integer":
        dist_template = IntegerDistribution
    elif dist_name == "datetime":
        dist_template = DatetimeDistribution
    else:
        dist_template = FloatDistribution

    return dist_template(name=feature, **dist_args)
