# third party
import pandas as pd
from pydantic import validate_arguments

# synthcity absolute
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

    def get(self, feature: str) -> Params:
        if feature not in self.domain:
            raise ValueError(f"invalid feature {feature}")

        return self.domain[feature]
