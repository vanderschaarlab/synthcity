# stdlib
from typing import Any, Dict, Generator, List

# third party
import numpy as np
import pandas as pd
from pydantic import BaseModel, validate_arguments, validator

# synthcity absolute
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
    constraint_to_distribution,
)


class Schema(BaseModel):
    """Utility class for defining the schema of a Dataset."""

    sampling_strategy: str = "marginal"  # uniform or marginal
    sequential_view: bool = False
    data: Any = None
    domain: Dict = {}

    @validator("domain", always=True)
    def _validate_domain(cls: Any, v: Any, values: Dict) -> Dict:
        if "data" not in values or values["data"] is None:
            return v

        feature_domain = {}
        raw = values["data"]
        protected_cols = ["seq_id", "seq_time_id"]
        sequential_view = values["sequential_view"]

        if isinstance(raw, DataLoader):
            if sequential_view:
                X, _ = raw.sequential_view()
            else:
                X = raw.dataframe()
        elif isinstance(raw, pd.DataFrame):
            X = raw
        else:
            raise ValueError("You need to provide a DataLoader in the data argument")

        if X.shape[1] == 0 or X.shape[0] == 0:
            return v

        sampling_strategy = values["sampling_strategy"]

        if sampling_strategy == "marginal":
            for col in X.columns:
                if col in protected_cols:
                    continue

                if X[col].dtype == "object" or len(X[col].unique()) < 10:
                    feature_domain[col] = CategoricalDistribution(name=col, data=X[col])
                elif X[col].dtype in ["int", "int32", "int64", "uint32", "uint64"]:
                    feature_domain[col] = IntegerDistribution(name=col, data=X[col])
                elif X[col].dtype in ["float", "float32", "float64", "double"]:
                    feature_domain[col] = FloatDistribution(name=col, data=X[col])
                else:
                    raise ValueError("unsupported format ", col)
        elif sampling_strategy == "uniform":
            for col in X.columns:
                if col in protected_cols:
                    continue

                if X[col].dtype == "object" or len(X[col].unique()) < 10:
                    feature_domain[col] = CategoricalDistribution(
                        name=col, choices=list(X[col].unique())
                    )
                elif X[col].dtype in ["int", "int32", "int64", "uint32", "uint64"]:
                    feature_domain[col] = IntegerDistribution(
                        name=col, low=X[col].min(), high=X[col].max()
                    )
                elif X[col].dtype in ["float", "float64", "double"]:
                    feature_domain[col] = FloatDistribution(
                        name=col, low=X[col].min(), high=X[col].max()
                    )
                else:
                    raise ValueError("unsupported format ", col)
        else:
            raise ValueError(f"invalid sampling strategy {sampling_strategy}")

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

    def adapt_dtypes(self, X: pd.DataFrame) -> pd.DataFrame:
        for feature in self.domain:
            if feature not in X.columns:
                continue
            X[feature] = X[feature].astype(
                self.domain[feature].dtype(), errors="ignore"
            )

        return X

    def as_constraints(self) -> Constraints:
        """Convert the schema to a list of Constraints."""
        constraints = Constraints(rules=[])
        for feature in self:
            constraints.extend(self[feature].as_constraint())

        return constraints

    @classmethod
    def from_constraints(
        cls, constraints: Constraints, sequential_view: bool = False
    ) -> "Schema":
        """Create a schema from a list of Constraints."""

        features = constraints.features()
        feature_domain: dict = {}

        for feature in features:
            dist = constraint_to_distribution(constraints, feature)
            feature_domain[feature] = dist

        return cls(sequential_view=sequential_view, domain=feature_domain)
