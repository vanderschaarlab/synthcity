# stdlib
from typing import Any, Dict, Generator, List

# third party
import pandas as pd
from pydantic import BaseModel, validate_arguments, validator

# synthcity absolute
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)


class Schema(BaseModel):
    """Utility class for defining the schema of a Dataset."""

    data: Any
    domain: Dict = {}

    @validator("domain", always=True)
    def _validate_domain(cls: Any, v: Any, values: Dict) -> Dict:
        feature_domain = {}
        if "data" not in values:
            raise ValueError("You need to provide the data argument")

        X = values["data"]
        if not isinstance(X, pd.DataFrame):
            raise ValueError("You need to provide a DataFrame in the data argument")

        for col in X.columns:
            if X[col].dtype == "object" or len(X[col].unique()) < 10:
                feature_domain[col] = CategoricalDistribution(
                    name=col, choices=list(X[col].unique())
                )
            elif X[col].dtype == "int":
                feature_domain[col] = IntegerDistribution(
                    name=col, low=X[col].min(), high=X[col].max()
                )
            elif X[col].dtype == "float":
                feature_domain[col] = FloatDistribution(
                    name=col, low=X[col].min(), high=X[col].max()
                )
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

    def as_constraint(self) -> Constraints:
        """Convert the schema to a list of Constraints."""
        constraints = Constraints(rules=[])
        for feature in self:
            constraints.extend(self[feature].as_constraint())

        return constraints
