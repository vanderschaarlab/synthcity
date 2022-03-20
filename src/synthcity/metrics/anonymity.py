# stdlib
from typing import Dict, List, Optional, Tuple

# third party
import pandas as pd
from pydantic import validate_arguments


class kAnonimity:
    def __init__(
        self,
        k: int = 10,
        categorical_limit: int = 5,
        max_partitions: Optional[int] = None,
    ) -> None:
        self.categorical_limit = categorical_limit
        self.k = k
        self.max_partitions = max_partitions

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _get_spans(
        self, X: pd.DataFrame, partition: pd.Index, scale: Optional[Dict] = None
    ) -> Dict:
        """Get the spans of all columns in the partition"""
        spans = {}
        for column in X.columns:
            if len(X[column].unique()) < self.categorical_limit:
                span = len(X[column][partition].unique())
            else:
                span = X[column][partition].max() - X[column][partition].min()
            if scale is not None:
                span = span / scale[column]
            spans[column] = span
        return spans

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _split(
        self, X: pd.DataFrame, partition: pd.Index, column: str, categoricals: list
    ) -> Tuple:
        """Return two partitions that split the given partition such that :
        * All rows with values of the column below the median are in one partition
        * All rows with values above or equal to the median are in the other."""
        Xpart = X[column][partition]
        if column in categoricals:
            values = Xpart.unique()
            lv = set(values[: len(values) // 2])
            rv = set(values[len(values) // 2 :])
            return Xpart.index[Xpart.isin(lv)], Xpart.index[Xpart.isin(rv)]
        else:
            median = Xpart.median()
            dfl = Xpart.index[Xpart < median]
            dfr = Xpart.index[Xpart >= median]
            return (dfl, dfr)

    def _get_categoricals(self, X: pd.DataFrame) -> List:
        categoricals = []
        for column in X.columns:
            if len(X[column].unique()) < self.categorical_limit:
                categoricals.append(column)

        return categoricals

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _partition_dataset(
        self,
        X: pd.DataFrame,
        feature_columns: List,
    ) -> List:
        """Learn a list of valid partitions that covers the entire dataframe."""
        finished_partitions = []
        partitions = [X.index]

        scale = self._get_spans(X, X.index)
        categoricals = self._get_categoricals(X)

        while partitions:
            partition = partitions.pop(0)
            spans = self._get_spans(X[feature_columns], partition, scale)
            for column, span in sorted(spans.items(), key=lambda x: -x[1]):
                lp, rp = self._split(X, partition, column, categoricals)
                if not self._is_index_k_anonymous(lp) or not self._is_index_k_anonymous(
                    rp
                ):
                    continue
                partitions.extend((lp, rp))
                break
            else:
                finished_partitions.append(partition)
        return finished_partitions

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _is_index_k_anonymous(self, partition: pd.Index) -> bool:
        """True if the partition is valid according to our k-anonymity criteria, False otherwise."""
        return len(partition) >= self.k

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def anonymize(
        self,
        X: pd.DataFrame,
        sensitive_column: str,
    ) -> pd.DataFrame:
        features = list(X.columns)
        features.remove(sensitive_column)

        partitions = self._partition_dataset(X, features)

        def agg_categorical_column(series: pd.Series) -> list:
            return [",".join([str(v) for v in set(series)])]

        def agg_numerical_column(series: pd.Series) -> list:
            return [series.mean()]

        categoricals = self._get_categoricals(X)

        rows = []
        for i, partition in enumerate(partitions):
            if self.max_partitions is not None and i > self.max_partitions:
                break

            X_part = X.loc[partition]

            X_part_cat = X_part[[c for c in X_part.columns if c in categoricals]]
            X_part_num = X_part[[c for c in X.columns if c not in categoricals]]

            grouped_columns = pd.concat(
                [
                    X_part_num.agg(agg_numerical_column),
                    X_part_cat.agg(agg_categorical_column),
                ],
                axis=1,
            )

            sensitive_counts = X_part.groupby(sensitive_column).agg(
                {sensitive_column: "count"}
            )
            values = grouped_columns.iloc[0].to_dict()
            for sensitive_value, count in sensitive_counts[sensitive_column].items():
                if count == 0:
                    continue
                values.update(
                    {
                        sensitive_column: sensitive_value,
                    }
                )
                for cnt in range(count):
                    rows.append(values.copy())

        result = pd.DataFrame(rows)
        if not self.is_k_anonymous(result, sensitive_column):
            raise RuntimeError("Dataset anonymize failed")

        return result

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def is_k_anonymous(self, X: pd.DataFrame, sensitive_column: str) -> bool:
        """True if the dataset is valid according to our k-anonymity criteria, False otherwise."""

        features = list(X.columns)
        features.remove(sensitive_column)
        X_groupby = X.groupby(features).size().reset_index(name="count")

        return bool(X_groupby["count"].min() >= self.k)
