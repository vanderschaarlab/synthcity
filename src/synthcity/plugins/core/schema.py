# stdlib
from typing import Generator

# third party
import pandas as pd
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.params import Categorical, Float, Integer, Params


class Schema:
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(self, X: pd.DataFrame) -> None:
        feature_domain = {}
        for col in X.columns:
            if X[col].dtype == "object" or len(X[col].unique()) < 10:
                feature_domain[col] = Categorical(col, X[col].unique())
            elif X[col].dtype == "int":
                feature_domain[col] = Integer(col, X[col].min(), X[col].max())
            elif X[col].dtype == "float":
                feature_domain[col] = Float(col, X[col].min(), X[col].max())
            else:
                raise RuntimeError("unsupported format ", col)

        self.domain = feature_domain

    def get(self, key: str) -> Params:
        if key not in self.domain:
            raise ValueError(f"invalid feature {key}")

        return self.domain[key]

    def __getitem__(self, key: str) -> Params:
        return self.get(key)

    def __iter__(self) -> Generator:
        for x in self.domain:
            yield x

    def __len__(self) -> int:
        return len(self.domain)

    def includes(self, other: "Schema") -> bool:
        for feature in other:
            if feature not in self.domain:
                return False

            if not self[feature].includes(other[feature]):
                return False

        return True

    def as_constraint(self) -> Constraints:
        constraints = Constraints([])
        for feature in self:
            constraints.extend(self[feature].as_constraint())

        return constraints
