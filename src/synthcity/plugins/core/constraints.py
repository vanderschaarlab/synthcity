# stdlib
from typing import Any, Dict

# third party
import pandas as pd
from pydantic import validate_arguments

OP_KEY = 0
THRESHOLD_KEY = 1


class Constraints:
    @validate_arguments
    def __init__(
        self,
        name: str,
        rules: Dict[str, tuple] = {},
    ) -> None:
        self.name = name
        self.rules = rules
        self.supported_ops = ["lt", "le", "gt", "ge", "eq"]

        for feature in self.rules:
            if (
                not isinstance(self.rules[feature], tuple)
                or len(self.rules[feature]) < 2
            ):
                raise ValueError(
                    f"Invalid constraint. Expecting tuple, but got {self.rules[feature]}"
                )

            op = self.rules[feature][OP_KEY]
            if op not in self.supported_ops:
                raise ValueError(
                    f"Invalid operation {op}. Supported ops: {self.supported_ops}"
                )

    def _eval(self, X: pd.DataFrame, feature: str, op: str, threshold: Any) -> bool:
        if op == "lt":
            return X[feature] < threshold
        elif op == "le":
            return X[feature] <= threshold
        elif op == "gt":
            return X[feature] > threshold
        elif op == "ge":
            return X[feature] >= threshold
        elif op == "eq":
            return X[feature] == threshold
        else:
            raise RuntimeError("unsupported operation", op)

    def match(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        res = pd.Series([True] * len(X), index=X.index)
        for feature in self.rules:
            res &= self._eval(
                X,
                feature,
                self.rules[feature][OP_KEY],
                self.rules[feature][THRESHOLD_KEY],
            )
        return X[res]
