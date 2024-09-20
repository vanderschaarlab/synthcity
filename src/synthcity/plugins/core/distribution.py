# stdlib
from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# third party
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("marginal_distribution", mode="before")
    def _validate_marginal_distribution(
        cls: Any, v: Any, values: ValidationInfo
    ) -> Dict:
        if "data" not in values.data or values.data["data"] is None:
            return v

        data = values.data["data"]
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
        """Get the min value of the distribution."""
        ...

    @abstractmethod
    def max(self) -> Any:
        """Get the max value of the distribution."""
        ...

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and self.get() == other.get()

    def __contains__(self, item: Any) -> bool:
        """
        Example:
        >>> dist = CategoricalDistribution(name="foo", choices=["a", "b", "c"])
        >>> "a" in dist
        True
        """
        return self.has(item)

    @abstractmethod
    def dtype(self) -> str:
        ...


class CategoricalDistribution(Distribution):
    """
    .. inheritance-diagram:: synthcity.plugins.core.distribution.CategoricalDistribution
        :parts: 1
    """

    choices: List = Field(default_factory=list)

    @field_validator("choices", mode="before")
    def _validate_choices(cls: Any, v: List, values: ValidationInfo) -> List:
        mkey = "marginal_distribution"
        # Check if marginal_distribution is present and return its index
        if mkey in values.data and values.data[mkey] is not None:
            return list(values.data[mkey].index)

        # If choices is empty, raise a ValueError
        if len(v) == 0:
            raise ValueError(
                "Invalid choices for CategoricalDistribution. Provide data or choices params"
            )
        # Ensure choices are unique and sorted
        return sorted(set(v))

    def get(self) -> List[Any]:
        return [self.name, self.choices]

    def sample(self, count: int = 1) -> Any:
        np.random.seed(self.random_state)
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

    low: float = Field(default=np.finfo(np.float64).min)
    high: float = Field(default=np.finfo(np.float64).max)

    @field_validator("low", mode="before")
    def _validate_low_thresh(cls: Any, v: float, values: ValidationInfo) -> float:
        mkey = "marginal_distribution"
        if mkey in values.data and values.data[mkey] is not None:
            return values.data[mkey].index.min()

        return v

    @field_validator("high", mode="before")
    def _validate_high_thresh(cls: Any, v: float, values: ValidationInfo) -> float:
        mkey = "marginal_distribution"
        if mkey in values.data and values.data[mkey] is not None:
            return values.data[mkey].index.max()

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

    def dtype(self) -> str:
        return "float"


class LogDistribution(FloatDistribution):
    low: float = np.finfo(np.float64).tiny
    high: float = np.finfo(np.float64).max

    def get(self) -> List[Any]:
        return [self.name, self.low, self.high]

    def sample(self, count: int = 1) -> Any:
        np.random.seed(self.random_state)
        msamples = self.sample_marginal(count)
        if msamples is not None:
            return msamples
        lo, hi = np.log2(self.low), np.log2(self.high)
        return 2.0 ** np.random.uniform(lo, hi, count)


class IntegerDistribution(Distribution):
    """
    .. inheritance-diagram:: synthcity.plugins.core.distribution.IntegerDistribution
        :parts: 1
    """

    low: int = Field(default=np.iinfo(np.int64).min)
    high: int = Field(default=np.iinfo(np.int64).max)
    step: int = Field(default=1)

    @field_validator("low", mode="before")
    def _validate_low_thresh(cls: Any, v: int, values: ValidationInfo) -> int:
        mkey = "marginal_distribution"
        # If marginal_distribution is present, use its minimum value as 'low'
        if mkey in values.data and values.data[mkey] is not None:
            return int(values.data[mkey].index.min())

        # Otherwise, return the value of 'low' field as is
        return v

    @field_validator("high", mode="before")
    def _validate_high_thresh(cls: Any, v: int, values: ValidationInfo) -> int:
        mkey = "marginal_distribution"
        # If marginal_distribution is present, use its maximum value as 'high'
        if mkey in values.data and values.data[mkey] is not None:
            return int(values.data[mkey].index.max())

        # Otherwise, return the value of 'high' field as is
        return v

    @field_validator("step", mode="before")
    def _validate_step(cls: Any, v: int, values: ValidationInfo) -> int:
        # Ensure that the step is greater than or equal to 1
        if v < 1:
            raise ValueError("Step must be greater than 0")

        # Return the validated step value
        return v

    def get(self) -> List[Any]:
        return [self.name, self.low, self.high, self.step]

    def sample(self, count: int = 1) -> Any:
        np.random.seed(self.random_state)
        msamples = self.sample_marginal(count)
        if msamples is not None:
            return msamples

        steps = (self.high - self.low) // self.step
        samples = np.random.choice(steps + 1, count)
        return samples * self.step + self.low

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

    def dtype(self) -> str:
        return "int"


class IntLogDistribution(IntegerDistribution):
    low: int = Field(default=1)
    high: int = Field(default=np.iinfo(np.int64).max)

    @field_validator("step", mode="before")
    def _validate_step(cls: Any, v: int, values: ValidationInfo) -> int:
        if v != 1:
            raise ValueError("Step must be 1 for IntLogDistribution")
        return v

    def get(self) -> List[Any]:
        return [self.name, self.low, self.high]

    def sample(self, count: int = 1) -> Any:
        np.random.seed(self.random_state)
        msamples = self.sample_marginal(count)
        if msamples is not None:
            return msamples
        lo, hi = np.log2(self.low), np.log2(self.high)
        samples = 2.0 ** np.random.uniform(lo, hi, count)
        return samples.astype(int)


class DatetimeDistribution(Distribution):
    """
    .. inheritance-diagram:: synthcity.plugins.core.distribution.DatetimeDistribution
        :parts: 1
    """

    low: datetime = Field(default_factory=lambda: datetime.utcfromtimestamp(0))
    high: datetime = Field(default_factory=lambda: datetime.now())
    step: timedelta = Field(default=timedelta(microseconds=1))
    offset: timedelta = Field(default=timedelta(seconds=120))

    @field_validator("low", mode="before")
    def _validate_low_thresh(cls: Any, v: datetime, values: ValidationInfo) -> datetime:
        mkey = "marginal_distribution"
        if mkey in values.data and values.data[mkey] is not None:
            v = values.data[mkey].index.min()
        return v

    @field_validator("high", mode="before")
    def _validate_high_thresh(
        cls: Any, v: datetime, values: ValidationInfo
    ) -> datetime:
        mkey = "marginal_distribution"
        if mkey in values.data and values.data[mkey] is not None:
            v = values.data[mkey].index.max()
        return v

    def get(self) -> List[Any]:
        return [self.name, self.low, self.high, self.step, self.offset]

    def sample(self, count: int = 1) -> Any:
        np.random.seed(self.random_state)
        msamples = self.sample_marginal(count)
        if msamples is not None:
            return msamples

        n = (self.high - self.low) // self.step + 1
        samples = np.round(np.random.rand(count) * n - 0.5)
        return self.low + samples * self.step

    def has(self, val: datetime) -> bool:
        return self.low <= val and val <= self.high

    def includes(self, other: "Distribution") -> bool:
        return (
            self.min() - self.offset <= other.min()
            and other.max() <= self.max() + self.offset
        )

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
