# stdlib
from typing import Any, Generator, List, Tuple

# third party
import pandas as pd
from pydantic import validate_arguments


class Constraints:
    @validate_arguments
    def __init__(
        self,
        rules: List[Tuple[str, str, Any]] = [],
    ) -> None:
        self.supported_ops = ["lt", "le", "gt", "ge", "eq", "in"]

        for rule in rules:
            if len(rule) < 3:
                raise ValueError(f"Invalid constraint. Expecting tuple, but got {rule}")

            feature, op, thresh = rule

            if op not in self.supported_ops:
                raise ValueError(
                    f"Invalid operation {op}. Supported ops: {self.supported_ops}"
                )
            if op in ["in"]:
                assert isinstance(thresh, list)
        self.rules = rules

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
        elif op == "in":
            return X[feature].isin(threshold)
        else:
            raise RuntimeError("unsupported operation", op)

    def filter(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        res = pd.Series([True] * len(X), index=X.index)
        for feature, op, thresh in self.rules:
            res &= self._eval(
                X,
                feature,
                op,
                thresh,
            )
        return res

    def match(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.filter(X)]

    def extend(self, other: "Constraints") -> "Constraints":
        self.rules.extend(other.rules)

        return self

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self) -> Generator:
        for x in self.rules:
            yield x
