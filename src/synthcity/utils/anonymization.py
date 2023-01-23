# stdlib
from typing import Dict, List, Optional, Tuple

# third party
import pandas as pd
from pydantic import validate_arguments
from sklearn.preprocessing import LabelEncoder

# synthcity absolute
from synthcity.metrics.eval_privacy import kAnonymization
from synthcity.plugins.core.dataloader import GenericDataLoader


class DatasetAnonymization:
    """Dataset Anonymization helper based on the k-Anonymization, l-Diversity and t-Closeness methods.

    k-Anonymity states that every individual in one dataset partition is indistinguishable from at least k - 1 other individuals.
    l-Diversity uses a stronger privacy definition and claims that every generalized block has to contain at least l different sensitive values.
    An equivalence class is said to have t-closeness if the distance between the distribution of a sensitive attribute in this class and the distribution of the attribute in the whole table is no more than a threshold t. A table is said to have t-closeness if all equivalence classes have t-closeness.
    For that, we measure the Kolmogorov-Smirnov distance between the empirical probability distribution of the sensitive attribute over the entire dataset vs. the distribution over the partition."""

    @validate_arguments
    def __init__(
        self,
        k_threshold: int = 10,
        l_diversity: int = 2,
        t_threshold: float = 0.2,
        categorical_limit: int = 5,
        max_partitions: Optional[int] = None,
    ) -> None:
        if k_threshold < 1:
            raise ValueError(
                f"Invalid value for k_threshold = {k_threshold}. Must be >= 1"
            )
        if l_diversity < 1:
            raise ValueError(
                f"Invalid value for l_threshold = {l_diversity}. Must be >= 1"
            )
        if t_threshold > 1:
            raise ValueError(
                f"Invalid value for t_threshold = {t_threshold}. Must be < 1"
            )
        self.k_threshold = k_threshold
        self.l_diversity = l_diversity
        self.t_threshold = t_threshold

        self.categorical_limit = categorical_limit
        self.max_partitions = max_partitions

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def is_anonymous(self, X: pd.DataFrame, sensitive_features: List[str] = []) -> bool:
        """True if the dataset is valid according to the k-anonymity criteria, False otherwise."""
        evaluator = kAnonymization()
        return bool(
            evaluator.evaluate_data(
                GenericDataLoader(X, sensitive_features=sensitive_features)
            )
            >= self.k_threshold
        )

    def _setup(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        encoders = {}

        for col in X.columns:
            if X[col].dtype != "object":
                continue

            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col])
            encoders[col] = encoder

        return X, encoders

    def _tear_down(self, X: pd.DataFrame, encoders: Dict) -> pd.DataFrame:
        for col in encoders:
            X[col] = encoders[col].inverse_transform(X[col])

        return X

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def anonymize(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        for column in X.columns:
            X = self.anonymize_column(X, column)

        return X

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def anonymize_columns(
        self,
        X: pd.DataFrame,
        sensitive_features: List[str],
    ) -> pd.DataFrame:
        for column in sensitive_features:
            X = self.anonymize_column(X, column)

        return X

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def anonymize_column(
        self,
        X: pd.DataFrame,
        sensitive_column: str,
    ) -> pd.DataFrame:
        X, encoders = self._setup(X)
        features = self._get_features(X, [sensitive_column])

        partitions = self._partition_dataset(X, features, sensitive_column)

        def agg_categorical_column(series: pd.Series) -> list:
            # TODO
            return [series.mean()]

        def agg_numerical_column(series: pd.Series) -> list:
            return [series.mean()]

        categoricals = self._get_categoricals(X)

        rows = []
        for i, partition in enumerate(partitions):
            if self.max_partitions is not None and i > self.max_partitions:
                break

            X_part = X.loc[partition]

            X_part_cat = X_part[[c for c in X_part.columns if c in categoricals]]
            X_part_num = X_part[[c for c in X_part.columns if c not in categoricals]]

            num_agg = X_part_num.agg(agg_numerical_column)
            cat_agg = X_part_cat.agg(agg_categorical_column)

            aggregations = []

            if len(num_agg) > 0:
                aggregations.append(num_agg)
            if len(cat_agg) > 0:
                aggregations.append(cat_agg)

            grouped_columns = pd.concat(
                aggregations,
                axis=1,
            )

            if set(grouped_columns.columns) != set(X_part.columns):
                raise RuntimeError(
                    f"Invalid columns {set(grouped_columns.columns)} != {set(X_part.columns)}"
                )
            grouped_columns = grouped_columns[X_part.columns]

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

        return self._tear_down(result, encoders)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _get_features(
        self, X: pd.DataFrame, sensitive_features: List[str] = []
    ) -> List:
        """Return the non-sensitive features from dataset X"""
        features = list(X.columns)
        for col in sensitive_features:
            if col in features:
                features.remove(col)

        return features

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

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
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
        sensitive_column: str,
    ) -> List:
        """Learn a list of valid partitions that covers the entire dataframe."""
        finished_partitions = []
        partitions = [X.index]

        scale = self._get_spans(X, X.index)
        categoricals = self._get_categoricals(X)
        freqs = self._get_frequencies(X, sensitive_column)

        while partitions:
            partition = partitions.pop(0)
            spans = self._get_spans(X[feature_columns], partition, scale)
            for column, span in sorted(spans.items(), key=lambda x: -x[1]):
                lp, rp = self._split(X, partition, column, categoricals)
                if not self._is_partition_anonymous(
                    X, lp, sensitive_column, freqs, categoricals
                ) or not self._is_partition_anonymous(
                    X, rp, sensitive_column, freqs, categoricals
                ):
                    continue
                partitions.extend((lp, rp))
                break
            else:
                finished_partitions.append(partition)
        return finished_partitions

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _get_frequencies(self, X: pd.DataFrame, sensitive_column: str) -> Dict:
        freq = {}
        total_count = float(len(X))
        group_counts = X.groupby(sensitive_column)[sensitive_column].agg("count")
        for value, count in group_counts.to_dict().items():
            p = count / total_count
            freq[value] = p

        return freq

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _is_k_anonymous(self, X: pd.DataFrame) -> bool:
        """True if the partition is valid according to the k-Anonymity criteria, False otherwise."""
        return len(X) >= self.k_threshold

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _is_l_diverse(self, X: pd.DataFrame, sensitive_column: str) -> bool:
        """True if the partition is valid according to the l-Diversity criteria, False otherwise."""
        return len(X[sensitive_column].unique()) >= self.l_diversity

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _t_closeness(self, X: pd.DataFrame, column: str, freqs: Dict) -> float:
        cnt = float(len(X))
        d_max = -1.0
        group_counts = X.groupby(column)[column].agg("count")
        for value, count in group_counts.to_dict().items():
            p = count / (cnt + 1e-8)
            d = abs(p - freqs[value])
            if d > d_max:
                d_max = d
        return d_max

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _is_t_close(
        self,
        X: pd.DataFrame,
        sensitive_column: str,
        frequencies: Dict,
        categorical_cols: List,
    ) -> bool:
        """True if the dataset is valid according to the t-closeness criteria, False otherwise."""
        if sensitive_column not in categorical_cols:
            return True

        return self._t_closeness(X, sensitive_column, frequencies) <= self.t_threshold

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _is_partition_anonymous(
        self,
        X: pd.DataFrame,
        partition: pd.Index,
        sensitive_column: str,
        frequencies: Dict = {},
        categorical_columns: List = [],
    ) -> bool:
        Xpartition = X.loc[partition]
        return (
            self._is_k_anonymous(Xpartition)
            and self._is_l_diverse(Xpartition, sensitive_column)
            and self._is_t_close(
                Xpartition, sensitive_column, frequencies, categorical_columns
            )
        )
