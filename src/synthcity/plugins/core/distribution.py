# stdlib
from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FieldValidationInfo,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

# synthcity absolute
from synthcity.plugins.core.constraints import Constraints

Rule = Tuple[str, str, Any]  # Define a type alias for clarity


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
    random_state: Optional[int] = None
    sampling_strategy: str = "marginal"
    _rng: np.random.Generator = PrivateAttr()
    # DP parameters
    marginal_distribution: Optional[pd.Series] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("marginal_distribution", mode="before")
    def _validate_marginal_distribution(
        cls: Any, v: Any, values: FieldValidationInfo
    ) -> Optional[pd.Series]:
        if "data" not in values.data or values.data["data"] is None:
            return v

        data = values.data["data"]
        if not isinstance(data, pd.Series):
            raise ValueError(f"Invalid data type {type(data)}")

        marginal = data.value_counts(dropna=False)
        del values["data"]

        return marginal

    @model_validator(mode="after")
    def initialize_rng(cls, model: "Distribution") -> "Distribution":
        """
        Initializes the random number generator after model validation.
        """
        if model.random_state is not None:
            model._rng = np.random.default_rng(model.random_state)
        else:
            model._rng = np.random.default_rng()
        return model

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

        return self._rng.choice(
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

    data: Optional[pd.Series] = None
    marginal_distribution: Optional[pd.Series] = None
    choices: List[Any] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_and_initialize(
        cls, model: "CategoricalDistribution"
    ) -> "CategoricalDistribution":
        """
        Validates and initializes choices and marginal_distribution based on data or provided choices.
        Ensures that choices are unique and sorted.
        """
        if model.data is not None:
            # Set marginal_distribution based on data
            model.marginal_distribution = model.data.value_counts(normalize=True)
            model.choices = model.marginal_distribution.index.tolist()
        elif model.choices is not None:
            # Ensure choices are unique and sorted
            model.choices = sorted(set(model.choices))
            # Set uniform probabilities
            probabilities = np.ones(len(model.choices)) / len(model.choices)
            model.marginal_distribution = pd.Series(probabilities, index=model.choices)
        else:
            raise ValueError(
                "Invalid CategoricalDistribution: Provide either 'data' or 'choices'."
            )

        # Additional validation to ensure consistency
        if not isinstance(model.choices, list) or len(model.choices) == 0:
            raise ValueError(
                "CategoricalDistribution must have a non-empty 'choices' list."
            )
        if not isinstance(model.marginal_distribution, pd.Series):
            raise ValueError(
                "CategoricalDistribution must have a valid 'marginal_distribution'."
            )
        if len(model.choices) != len(model.marginal_distribution):
            raise ValueError(
                "'choices' and 'marginal_distribution' must have the same length."
            )

        return model

    def sample(self, count: int = 1) -> Any:
        """
        Samples values from the distribution based on the specified sampling strategy.
        If the distribution has only one choice, returns an array filled with that value.
        """
        if self.choices is not None and len(self.choices) == 1:
            samples = np.full(count, self.choices[0])
        else:
            if self.sampling_strategy == "marginal":
                if self.marginal_distribution is None:
                    raise ValueError(
                        "Cannot sample based on marginal distribution: marginal_distribution is not provided."
                    )
                return self._rng.choice(
                    self.marginal_distribution.index,
                    size=count,
                    p=self.marginal_distribution.values,
                )
            elif self.sampling_strategy == "uniform":
                return self._rng.choice(self.choices, size=count)
            else:
                raise ValueError(
                    f"Unsupported sampling strategy '{self.sampling_strategy}'."
                )
        return samples

    def get(self) -> List[Any]:
        """
        Returns the metadata of the distribution.
        """
        return [self.name, self.choices]

    def has(self, val: Any) -> bool:
        """
        Checks if a value is among the distribution's choices.
        """
        return val in self.choices

    def includes(self, other: "Distribution") -> bool:
        """
        Checks if another categorical distribution's choices are a subset of this distribution's choices.
        """
        if not isinstance(other, CategoricalDistribution):
            return False
        return set(other.choices).issubset(set(self.choices))

    def as_constraint(self) -> Constraints:
        """
        Converts the distribution to a set of constraints.
        """
        return Constraints(rules=[(self.name, "in", list(self.choices))])

    def min(self) -> Any:
        """
        Returns the minimum value among the choices.
        """
        return min(self.choices)

    def max(self) -> Any:
        """
        Returns the maximum value among the choices.
        """
        return max(self.choices)

    def dtype(self) -> str:
        """
        Determines the data type based on the choices.
        """
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

    low: Optional[float] = Field(default=None)
    high: Optional[float] = Field(default=None)
    _is_constant: bool = PrivateAttr(False)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_and_initialize(cls, model: "FloatDistribution") -> "FloatDistribution":
        """
        Validates and initializes the distribution.
        Sets '_is_constant' based on whether 'low' equals 'high'.
        Initializes 'marginal_distribution' based on 'data' if provided.
        """
        if model.data is not None:
            # Initialize marginal_distribution based on data
            # For float data, use value_counts(normalize=True) if data has repeated values
            # This will create a discrete approximation of the distribution
            model.marginal_distribution = model.data.value_counts(
                normalize=True
            ).sort_index()
            model.low = float(model.data.min())
            model.high = float(model.data.max())
        elif model.marginal_distribution is not None:
            # Set 'low' and 'high' based on marginal_distribution
            model.low = float(model.marginal_distribution.index.min())
            model.high = float(model.marginal_distribution.index.max())
        else:
            # Ensure 'low' and 'high' are provided
            if model.low is None or model.high is None:
                raise ValueError(
                    "FloatDistribution requires 'low' and 'high' values if 'data' or 'marginal_distribution' is not provided."
                )

        # Validate that low <= high
        if model.low > model.high:
            raise ValueError(
                f"Invalid range for '{model.name}': low ({model.low}) cannot be greater than high ({model.high})."
            )

        # Set _is_constant based on low == high
        model._is_constant = model.low == model.high

        # Ensure that low and high are finite numbers
        if not np.isfinite(model.low) or not np.isfinite(model.high):
            raise ValueError(
                f"Invalid range for '{model.name}': low or high is not finite (low={model.low}, high={model.high})."
            )

        return model

    def sample(self, count: int = 1) -> Any:
        """
        Samples values from the distribution.
        If the distribution is constant, returns an array filled with the constant value.
        Otherwise, samples based on the marginal distribution or uniform sampling.
        """
        if self._is_constant:
            if self.low is None:
                raise ValueError(
                    "Cannot sample: 'low' is None for a constant distribution."
                )
            samples = np.full(count, self.low)
        else:
            if self.low is None or self.high is None:
                raise ValueError("Cannot sample: 'low' or 'high' is None.")
            if (
                self.sampling_strategy == "marginal"
                and self.marginal_distribution is not None
            ):
                # Sample based on marginal distribution
                return self._rng.choice(
                    self.marginal_distribution.index.values,
                    size=count,
                    p=self.marginal_distribution.values,
                )
            else:
                # Proceed with uniform sampling
                samples = self._rng.uniform(low=self.low, high=self.high, size=count)
        return samples

    def get(self) -> List[Any]:
        """
        Returns the metadata of the distribution.
        """
        return [self.name, self.low, self.high]

    def has(self, val: Any) -> bool:
        """
        Checks if a value is within the distribution's range.
        """
        return self.low <= val <= self.high

    def includes(self, other: "Distribution") -> bool:
        """
        Checks if another distribution is entirely within this distribution.
        """
        if self.min() is None or self.max() is None:
            return False
        if other.min() is None or other.max() is None:
            return False
        return self.min() <= other.min() and other.max() <= self.max()

    def as_constraint(self) -> Constraints:
        """
        Converts the distribution to a set of constraints.
        """
        return Constraints(
            rules=[
                (self.name, "le", self.high),
                (self.name, "ge", self.low),
                (self.name, "dtype", "float"),
            ]
        )

    def min(self) -> Any:
        """
        Returns the minimum value of the distribution.
        """
        return self.low

    def max(self) -> Any:
        """
        Returns the maximum value of the distribution.
        """
        return self.high

    def dtype(self) -> str:
        """
        Returns the data type of the distribution.
        """
        return "float"


class LogDistribution(FloatDistribution):
    low: float = np.finfo(np.float64).tiny
    high: float = np.finfo(np.float64).max

    def get(self) -> List[Any]:
        return [self.name, self.low, self.high]

    def sample(self, count: int = 1) -> Any:
        msamples = self.sample_marginal(count)
        if msamples is not None:
            return msamples
        lo, hi = np.log2(self.low), np.log2(self.high)
        return 2.0 ** self._rng.uniform(lo, hi, count)


class IntegerDistribution(Distribution):
    """
    .. inheritance-diagram:: synthcity.plugins.core.distribution.IntegerDistribution
        :parts: 1
    """

    low: Optional[int] = Field(default=None)
    high: Optional[int] = Field(default=None)
    step: int = Field(default=1)
    _is_constant: bool = PrivateAttr(False)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_and_initialize(
        cls, model: "IntegerDistribution"
    ) -> "IntegerDistribution":
        """
        Validates and initializes the distribution.
        Sets '_is_constant' based on whether 'low' equals 'high'.
        Initializes 'marginal_distribution' based on 'data' if provided.
        """
        if model.data is not None:
            # Initialize marginal_distribution based on data
            model.marginal_distribution = model.data.value_counts(
                normalize=True
            ).sort_index()
            model.low = int(model.data.min())
            model.high = int(model.data.max())
        elif model.marginal_distribution is not None:
            # Infer 'low' and 'high' from the marginal distribution's index
            model.low = int(model.marginal_distribution.index.min())
            model.high = int(model.marginal_distribution.index.max())
        else:
            # Ensure 'low' and 'high' are provided
            if model.low is None or model.high is None:
                raise ValueError(
                    "IntegerDistribution requires 'low' and 'high' values if 'data' or 'marginal_distribution' is not provided."
                )

        # Validate that low <= high
        if model.low > model.high:
            raise ValueError(
                f"Invalid range for '{model.name}': low ({model.low}) cannot be greater than high ({model.high})."
            )

        # Set _is_constant based on low == high
        model._is_constant = model.low == model.high

        # Ensure that low and high are finite integers
        if not np.isfinite(model.low) or not np.isfinite(model.high):
            raise ValueError(
                f"Invalid range for '{model.name}': low or high is not finite (low={model.low}, high={model.high})."
            )

        # Ensure that 'step' is a positive integer
        if model.step <= 0:
            raise ValueError("'step' must be a positive integer.")

        # Adjust 'low' and 'high' to be compatible with 'step'
        model.low = model.low - ((model.low - (model.low % model.step)) % model.step)
        model.high = model.high - (
            (model.high - (model.high % model.step)) % model.step
        )

        # Re-validate after adjustment
        if model.low > model.high:
            raise ValueError(
                f"After adjusting with step, invalid range for '{model.name}': low ({model.low}) cannot be greater than high ({model.high})."
            )

        return model

    def sample(self, count: int = 1) -> Any:
        """
        Samples values from the distribution.
        If the distribution is constant, returns an array filled with the constant value.
        Otherwise, samples based on the marginal distribution or uniform sampling.
        """
        if self._is_constant:
            if self.low is None:
                raise ValueError(
                    "Cannot sample: 'low' is None for a constant distribution."
                )
            samples = np.full(count, self.low)
        else:
            if self.low is None or self.high is None:
                raise ValueError("Cannot sample: 'low' or 'high' is None.")
            if (
                self.sampling_strategy == "marginal"
                and self.marginal_distribution is not None
            ):
                # Sample based on marginal distribution
                return self._rng.choice(
                    self.marginal_distribution.index,
                    size=count,
                    p=self.marginal_distribution.values,
                )
            else:
                if self.low is None or self.high is None:
                    raise ValueError(
                        "Cannot sample based on uniform distribution: low or high is not provided."
                    )
                # Proceed with uniform sampling
                possible_values = np.arange(self.low, self.high + 1, self.step)
                samples = self._rng.choice(possible_values, size=count)
        return samples

    def get(self) -> List[Any]:
        """
        Returns the metadata of the distribution.
        """
        return [self.name, self.low, self.high, self.step]

    def has(self, val: Any) -> bool:
        """
        Checks if a value is within the distribution's range.
        """
        return self.low <= val <= self.high

    def includes(self, other: "Distribution") -> bool:
        """
        Checks if another distribution is entirely within this distribution.
        """
        if self.min() is None or self.max() is None:
            return False
        if other.min() is None or other.max() is None:
            return False
        return self.min() <= other.min() and other.max() <= self.max()

    def as_constraint(self) -> Constraints:
        """
        Converts the distribution to a set of constraints.
        """
        rules: List[Rule] = []
        if self.low is not None:
            rules.append((self.name, "ge", self.low))
        if self.high is not None:
            rules.append((self.name, "le", self.high))
        rules.append((self.name, "dtype", "int"))
        return Constraints(rules=rules)

    def min(self) -> Any:
        """
        Returns the minimum value of the distribution.
        """
        return self.low

    def max(self) -> Any:
        """
        Returns the maximum value of the distribution.
        """
        return self.high

    def dtype(self) -> str:
        """
        Returns the data type of the distribution.
        """
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
        msamples = self.sample_marginal(count)
        if msamples is not None:
            return msamples
        lo, hi = np.log2(self.low), np.log2(self.high)
        samples = 2.0 ** self._rng.uniform(lo, hi, count)
        return samples.astype(int)


class DatetimeDistribution(Distribution):
    """
    .. inheritance-diagram:: synthcity.plugins.core.distribution.DatetimeDistribution
        :parts: 1
    """

    low: Optional[datetime] = Field(default=None)
    high: Optional[datetime] = Field(default=None)
    step: timedelta = Field(default=timedelta(microseconds=1))
    offset: timedelta = Field(default=timedelta(seconds=120))
    _is_constant: bool = PrivateAttr(False)  # Correctly named with leading underscore

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_low_high(cls, model: "DatetimeDistribution") -> "DatetimeDistribution":
        """
        Validates that 'low' is less than or equal to 'high'.
        Sets '_is_constant' based on whether 'low' equals 'high'.
        """
        if model.marginal_distribution is not None:
            # Infer 'low' and 'high' from the marginal distribution's index
            model.low = model.marginal_distribution.index.min()
            model.high = model.marginal_distribution.index.max()
        else:
            # If 'marginal_distribution' is not provided, ensure 'low' and 'high' are set
            if model.low is None or model.high is None:
                if model.data is not None:
                    model.low = model.data.min()
                    model.high = model.data.max()
                else:
                    # Set default finite datetime values if not provided
                    model.low = datetime.fromtimestamp(0, timezone.utc)
                    model.high = datetime.now()
        if model.low is None or model.high is None:
            raise ValueError(
                "DatetimeDistribution requires 'low' and 'high' values if 'data' or 'marginal_distribution' is not provided."
            )
        # Validate that low <= high
        if model.low > model.high:
            raise ValueError(
                f"Invalid range for {model.name}: low ({model.low}) cannot be greater than high ({model.high})."
            )

        # Set _is_constant based on low == high
        model._is_constant = model.low == model.high

        # Ensure that low and high are valid datetime objects
        if not isinstance(model.low, datetime) or not isinstance(model.high, datetime):
            raise ValueError(
                f"Invalid range for {model.name}: low or high is not a valid datetime object (low={model.low}, high={model.high})."
            )

        # Ensure that 'step' is positive and non-zero
        if model.step.total_seconds() <= 0:
            raise ValueError("'step' must be a positive timedelta.")

        return model

    def sample(self, count: int = 1) -> Any:
        """
        Samples datetime values from the distribution.
        If the distribution is constant, returns a list filled with the constant datetime value.
        Otherwise, samples based on the specified sampling strategy.
        """
        if self._is_constant:
            if self.low is None:
                raise ValueError(
                    "Cannot sample constant datetime distribution: low is not provided."
                )
            samples = [self.low for _ in range(count)]
        else:
            if self.low is None or self.high is None:
                raise ValueError(
                    "Cannot sample datetime distribution: low or high is not provided."
                )
            if self.sampling_strategy in ["marginal", "uniform"]:
                msamples = self.sample_marginal(count)
                if msamples is not None:
                    return msamples
                if self.low is None or self.high is None:
                    raise ValueError(
                        "Cannot sample based on marginal distribution: low or high is not provided."
                    )
                total_seconds = (self.high - self.low).total_seconds()
                step_seconds = self.step.total_seconds()
                steps = int(total_seconds / step_seconds)
                step_indices = self._rng.integers(0, steps + 1, count)
                samples = [self.low + self.step * int(s) for s in step_indices]
            else:
                raise ValueError(
                    f"Unsupported sampling strategy '{self.sampling_strategy}'."
                )
        return samples

    def get(self) -> List[Any]:
        """
        Returns the metadata of the distribution.
        """
        return [self.name, self.low, self.high, self.step, self.offset]

    def has(self, val: datetime) -> bool:
        """
        Checks if a datetime value is within the distribution's range.
        """
        if self.low is None or self.high is None:
            raise ValueError("Cannot determine 'has' because 'low' or 'high' is None.")
        return self.low <= val <= self.high

    def includes(self, other: "Distribution") -> bool:
        """
        Checks if another datetime distribution is entirely within this distribution, considering the offset.
        """
        if self.low is None or self.high is None:
            return False
        if other.min() is None or other.max() is None:
            return False
        return (
            self.low - self.offset <= other.min()
            and other.max() <= self.high + self.offset
        )

    def as_constraint(self) -> Constraints:
        """
        Converts the distribution to a set of constraints.
        """
        return Constraints(
            rules=[
                (self.name, "le", self.high),
                (self.name, "ge", self.low),
                (self.name, "dtype", "datetime"),
            ]
        )

    def min(self) -> Optional[datetime]:
        """
        Returns the minimum datetime value of the distribution.
        """
        return self.low

    def max(self) -> Optional[datetime]:
        """
        Returns the maximum datetime value of the distribution.
        """
        return self.high

    def dtype(self) -> str:
        """
        Returns the data type of the distribution.
        """
        return "datetime"


class PassThroughDistribution(Distribution):
    """
    .. inheritance-diagram:: synthcity.plugins.core.distribution.PassThroughDistribution
        :parts: 1
    """

    data: pd.Series
    _dtype: str = PrivateAttr("")

    def setup_distribution(self) -> None:
        if self.data is None:
            raise ValueError("'data' must be provided for PassThroughDistribution.")

        # No additional attributes to set up since 'data' is used directly
        # Optionally, store the data type for dtype method
        self._dtype = str(self.data.dtype)

    def sample(self, count: int = 1) -> Any:
        msamples = self.sample_marginal(count)
        if msamples is not None:
            return msamples
        return self.data.sample(
            n=count, replace=True, random_state=self.random_state
        ).values

    def as_constraint(self) -> Constraints:
        # No constraints needed for pass-through columns
        return Constraints(rules=[])

    def get(self) -> List[Any]:
        # Return the unique values or any relevant info
        return [self.name]

    def has(self, val: Any) -> bool:
        # Check if the value exists in the data
        return val in self.data.values

    def includes(self, other: "Distribution") -> bool:
        # Since we are passing through values, we can define includes as checking if all values in other are in self.data
        if isinstance(other, PassThroughDistribution):
            return set(other.data.unique()).issubset(set(self.data.unique()))
        else:
            return False

    def min(self) -> Any:
        return self.data.min()

    def max(self) -> Any:
        return self.data.max()

    def dtype(self) -> str:
        return str(self.data.dtype)


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
