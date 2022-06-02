# stdlib
from typing import Any, Generator, List, Tuple

# third party
import numpy as np
import pandas as pd
from pydantic import BaseModel, validate_arguments, validator


class Constraints(BaseModel):
    rules: list = []

    @validator("rules")
    def _validate_rules(cls: Any, rules: List, values: dict, **kwargs: Any) -> List:
        supported_ops: list = [
            "<",
            ">=",
            "<=",
            ">",
            "==",
            "lt",
            "le",
            "gt",
            "ge",
            "eq",
            "in",
            "dtype",
        ]

        for rule in rules:
            if len(rule) < 3:
                raise ValueError(f"Invalid constraint. Expecting tuple, but got {rule}")

            feature, op, thresh = rule

            if op not in supported_ops:
                raise ValueError(
                    f"Invalid operation {op}. Supported ops: {supported_ops}"
                )
            if op in ["in"]:
                assert isinstance(thresh, list)
            elif op in ["dtype"]:
                assert isinstance(thresh, str)
        return rules

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _eval(self, X: pd.DataFrame, feature: str, op: str, operand: Any) -> pd.Index:
        """Evaluation primitive.

        Args:
            X: DataFrame. The dataset to apply the constraint on.
            feature: str. The column in the dataset to apply the constraint on.
            op: str. The operation to execute for the constraint.
            operand: Any. The operand for the binary operation.

        Returns:
            The pandas.Index which matches the constraint.
        """
        if op == "lt" or op == "<":
            return X[feature] < operand
        elif op == "le" or op == "<=":
            return X[feature] <= operand
        elif op == "gt" or op == ">":
            return X[feature] > operand
        elif op == "ge" or op == ">=":
            return X[feature] >= operand
        elif op == "eq" or op == "==":
            return X[feature] == operand
        elif op == "in":
            return X[feature].isin(operand)
        elif op == "dtype":
            return X[feature].dtype == operand
        else:
            raise RuntimeError("unsupported operation", op)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def filter(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the constraints to a DataFrame X.

        Args:
            X: DataFrame. The dataset to apply the constraints on.

        Returns:
            pandas.Index which matches all the constraints
        """
        X = pd.DataFrame(X)
        res = pd.Series([True] * len(X), index=X.index)
        for feature, op, thresh in self.rules:
            if feature not in X:
                res &= False
                continue

            prev = res.sum()
            res &= self._eval(
                X,
                feature,
                op,
                thresh,
            )
            if res.sum() == 0 and prev != 0:
                print(
                    "quality loss for constraints ",
                    feature,
                    op,
                    thresh,
                    X[feature].dtype,
                )
        return res

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def match(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the constraints to a DataFrame X and return the filtered dataset.

        Args:
            X: DataFrame. The dataset to apply the constraints on.

        Returns:
            The filtered Dataframe
        """

        return X[self.filter(X)]

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def is_valid(self, X: pd.DataFrame) -> bool:
        """Checks if all the rows in X meet the constraints.

        Args:
            X: DataFrame. The dataset to apply the constraints on.

        Returns:
            True if all rows match the constraints, False otherwise
        """

        return self.filter(X).sum() == len(X)

    def extend(self, other: "Constraints") -> "Constraints":
        """Extend the local constraints with more constraints.

        Args:
            other: The new constraints to add.

        Returns:
            self with the updated constraints.
        """
        self.rules.extend(other.rules)

        return self

    def __len__(self) -> int:
        """The number of constraint rules."""
        return len(self.rules)

    def __iter__(self) -> Generator:
        """Iterate the constraint rules."""
        for x in self.rules:
            yield x

    def features(self) -> List:
        results = []
        for feature, _, _ in self.rules:
            results.append(feature)

        return list(set(results))

    def feature_constraints(self, ref_feature: str) -> List:
        results = []
        for feature, op, threshold in self.rules:
            if feature != ref_feature:
                continue
            results.append((op, threshold))

        return results

    def feature_params(self, feature: str) -> Tuple:
        rules = self.feature_constraints(feature)

        dist_template = "float"
        dist_args = {"low": np.iinfo(np.int32).min, "high": np.iinfo(np.int32).max}

        for op, value in rules:
            if op == "in":
                dist_template = "categorical"
                if "choices" not in dist_args:
                    dist_args["choices"] = value
                    continue
                dist_args["choices"] = [v for v in value if v in dist_args["choices"]]

            elif op == "dtype" and value == "int":
                dist_template = "integer"
            elif (op == "le" or op == "<=") and value < dist_args["high"]:
                dist_args["high"] = value
            elif (op == "lt" or op == "<") and value < dist_args["high"]:
                dist_args["high"] = value - 1
            elif (op == "ge" or op == ">=") and dist_args["low"] < value:
                dist_args["low"] = value
            elif (op == "gt" or op == ">") and dist_args["low"] < value:
                dist_args["low"] = value + 1
            elif op == "eq" or op == "==":
                dist_args["low"] = value
                dist_args["high"] = value

        return dist_template, dist_args
