# stdlib
from typing import Any, Dict, Generator, List

# third party
import numpy as np
import pandas as pd
from pydantic import BaseModel, validate_arguments, validator

# synthcity absolute
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
    constraint_to_distribution,
)


class Schema(BaseModel):
    """Utility class for defining the schema of a Dataset."""

    data: Any = None
    domain: Dict = {}

    @validator("domain", always=True)
    def _validate_domain(cls: Any, v: Any, values: Dict) -> Dict:
        if "data" not in values or values["data"] is None:
            return v

        feature_domain = {}
        X = values["data"]
        if not isinstance(X, pd.DataFrame):
            raise ValueError("You need to provide a DataFrame in the data argument")

        for col in X.columns:
            if X[col].dtype == "object" or len(X[col].unique()) < 10:
                feature_domain[col] = CategoricalDistribution(name=col, data=X[col])
            elif X[col].dtype == "int":
                feature_domain[col] = IntegerDistribution(name=col, data=X[col])
            elif X[col].dtype == "float":
                feature_domain[col] = FloatDistribution(name=col, data=X[col])
            else:
                raise ValueError("unsupported format ", col)

        del values["data"]

        return feature_domain

    @validate_arguments
    def get(self, feature: str) -> Distribution:
        """Get the Distribution of a feature.

        Args:
            feature: str. the feature name

        Returns:
            The feature distribution
        """
        if feature not in self.domain:
            raise ValueError(f"invalid feature {feature}")

        return self.domain[feature]

    @validate_arguments
    def __getitem__(self, key: str) -> Distribution:
        """Get the Distribution of a feature.

        Args:
            feature: str. the feature name

        Returns:
            The feature distribution
        """
        return self.get(key)

    def __iter__(self) -> Generator:
        """Iterate the features distribution"""
        for x in self.domain:
            yield x

    def __len__(self) -> int:
        """Get the number of features"""
        return len(self.domain)

    def includes(self, other: "Schema") -> bool:
        """Test if another schema is included in the local one."""
        for feature in other:
            if feature not in self.domain:
                return False

            if not self[feature].includes(other[feature]):
                return False

        return True

    def features(self) -> List:
        return list(self.domain.keys())

    def sample(self, count: int) -> pd.DataFrame:
        samples = pd.DataFrame(
            np.zeros((count, len(self.features()))), columns=self.features()
        )

        for feature in self.features():
            samples[feature] = self.domain[feature].sample(count)

        return samples

    def as_constraints(self) -> Constraints:
        """Convert the schema to a list of Constraints."""
        constraints = Constraints(rules=[])
        for feature in self:
            constraints.extend(self[feature].as_constraint())

        return constraints

    @classmethod
    def from_constraints(cls, constraints: Constraints) -> "Schema":
        """Create a schema from a list of Constraints."""

        features = constraints.features()
        feature_domain: dict = {}

        for feature in features:
            dist = constraint_to_distribution(constraints, feature)
            feature_domain[feature] = dist

        return cls(domain=feature_domain)
