# third party
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes

# synthcity absolute
from synthcity.metrics.eval_privacy import kAnonymization, lDiversityDistinct
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils.anonymization import DatasetAnonymization


def test_k_anonymity_sanity() -> None:
    evaluator = DatasetAnonymization(
        k_threshold=44,
        categorical_limit=11,
        max_partitions=22,
    )

    assert evaluator.categorical_limit == 11
    assert evaluator.k_threshold == 44
    assert evaluator.max_partitions == 22


def test_k_anonymity_get_spans() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    evaluator = DatasetAnonymization()

    spans = evaluator._get_spans(X, X.index)

    assert set(spans.keys()) == set(X.columns)


def test_k_anonymity_split() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    evaluator = DatasetAnonymization()

    categoricals = evaluator._get_categoricals(X)
    for col in X.columns:
        lhs, rhs = evaluator._split(X, X.index, col, categoricals)

        assert len(lhs) > 0
        assert len(rhs) > 0
        assert len(lhs) + len(rhs) == len(X)

        assert evaluator._is_partition_anonymous(X, lhs, X.columns[0]) is True
        assert evaluator._is_partition_anonymous(X, rhs, X.columns[0]) is True


def test_k_anonymity_partition() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    evaluator = DatasetAnonymization(k_threshold=100)
    partitions = evaluator._partition_dataset(X, list(X.columns), X.columns[0])
    less_parts = len(partitions)

    evaluator = DatasetAnonymization(k_threshold=5)
    partitions = evaluator._partition_dataset(X, list(X.columns), X.columns[0])
    more_parts = len(partitions)

    assert less_parts > 0
    assert more_parts > 0
    assert less_parts < more_parts


def test_k_anonymity_validation() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    evaluator = DatasetAnonymization(k_threshold=1)
    assert evaluator.is_anonymous(X) is True

    evaluator = DatasetAnonymization(k_threshold=3)
    assert evaluator.is_anonymous(X) is False


def test_k_anonymity_get_categoricals() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    evaluator = DatasetAnonymization()

    assert evaluator._get_categoricals(X) == ["sex"]


def test_k_anonymity_get_freqs() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    evaluator = DatasetAnonymization()

    col = "sex"
    freqs = evaluator._get_frequencies(X, col)
    for val in X[col].unique():
        assert val in freqs


def test_k_anonymity_test_partition() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    evaluator = DatasetAnonymization()

    assert evaluator._is_k_anonymous(X) is True
    assert evaluator._is_k_anonymous(X[X["sex"] == 1234]) is False


def test_k_anonymity_anonymize_column() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    sensitive_features = ["target"]

    kevaluator = kAnonymization()
    k = kevaluator.evaluate_data(
        GenericDataLoader(X, sensitive_features=sensitive_features)
    )
    evaluator = DatasetAnonymization(k_threshold=k + 1)
    assert evaluator.is_anonymous(X, sensitive_features) is False

    anon_df = evaluator.anonymize_column(X, sensitive_column=sensitive_features[0])

    assert evaluator.is_anonymous(anon_df, sensitive_features) is True
    assert len(anon_df.drop_duplicates()) > 1


def test_k_anonymity_anonymize_columns() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    sensitive_features = list(X.columns)[:2]
    kevaluator = kAnonymization()
    k = kevaluator.evaluate_data(
        GenericDataLoader(X, sensitive_features=sensitive_features)
    )

    evaluator = DatasetAnonymization(k_threshold=k + 1)
    assert evaluator.is_anonymous(X, sensitive_features) is False

    anon_df = evaluator.anonymize_columns(X, sensitive_features=sensitive_features)

    assert evaluator.is_anonymous(anon_df, sensitive_features) is True
    assert len(anon_df.drop_duplicates()) > 1


def test_k_anonymity_anonymize_full() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    kevaluator = kAnonymization()
    k = kevaluator.evaluate_data(GenericDataLoader(X))
    evaluator = DatasetAnonymization(k_threshold=k + 1)
    assert evaluator.is_anonymous(X) is False

    anon_df = evaluator.anonymize(X)

    assert evaluator.is_anonymous(anon_df) is True


def test_l_diversity_sanity() -> None:
    evaluator = DatasetAnonymization(
        k_threshold=44,
        l_diversity=33,
        categorical_limit=11,
        max_partitions=22,
    )

    assert evaluator.categorical_limit == 11
    assert evaluator.k_threshold == 44
    assert evaluator.l_diversity == 33
    assert evaluator.max_partitions == 22


def test_l_diversity_get_spans() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    evaluator = DatasetAnonymization(l_diversity=20)

    spans = evaluator._get_spans(X, X.index)

    assert set(spans.keys()) == set(X.columns)


def test_l_diversity_split() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    evaluator = DatasetAnonymization()

    categoricals = evaluator._get_categoricals(X)
    for col in X.columns:
        lhs, rhs = evaluator._split(X, X.index, col, categoricals)

        assert len(lhs) > 0
        assert len(rhs) > 0
        assert len(lhs) + len(rhs) == len(X)

        assert evaluator._is_partition_anonymous(X, lhs, X.columns[0]) is True
        assert evaluator._is_partition_anonymous(X, rhs, X.columns[0]) is True


def test_l_diversity_partition() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    evaluator = DatasetAnonymization(l_diversity=20)
    partitions = evaluator._partition_dataset(X, list(X.columns), X.columns[0])
    less_parts = len(partitions)

    evaluator = DatasetAnonymization(l_diversity=1)
    partitions = evaluator._partition_dataset(X, list(X.columns), X.columns[0])
    more_parts = len(partitions)

    assert less_parts > 0
    assert more_parts > 0
    assert less_parts < more_parts


def test_l_diversity_validation() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    evaluator = DatasetAnonymization(k_threshold=1, l_diversity=1)
    assert evaluator.is_anonymous(X, sensitive_features=[X.columns[0]]) is True

    evaluator = DatasetAnonymization(k_threshold=2, l_diversity=1)
    assert evaluator.is_anonymous(X, sensitive_features=[X.columns[0]]) is False


def test_l_diversity_anonymize_column() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    sensitive_features = ["target"]
    kevaluator = kAnonymization()
    k_threshold = kevaluator.evaluate_data(
        GenericDataLoader(X, sensitive_features=sensitive_features)
    )

    levaluator = lDiversityDistinct()
    l_diversity = levaluator.evaluate_data(
        GenericDataLoader(X, sensitive_features=sensitive_features)
    )

    evaluator = DatasetAnonymization(
        k_threshold=k_threshold + 1, l_diversity=l_diversity + 1
    )
    assert evaluator.is_anonymous(X, sensitive_features) is False

    anon_df = evaluator.anonymize_column(X, sensitive_column=sensitive_features[0])

    assert evaluator.is_anonymous(anon_df, sensitive_features) is True
    assert len(anon_df.drop_duplicates()) > 1


def test_l_diversity_anonymize_columns() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    sensitive_features = list(X.columns)[:2]
    kevaluator = kAnonymization()
    k_threshold = kevaluator.evaluate_data(
        GenericDataLoader(X, sensitive_features=sensitive_features)
    )

    levaluator = lDiversityDistinct()
    l_diversity = levaluator.evaluate_data(
        GenericDataLoader(X, sensitive_features=sensitive_features)
    )

    evaluator = DatasetAnonymization(
        k_threshold=k_threshold + 1, l_diversity=l_diversity + 1
    )
    assert evaluator.is_anonymous(X, sensitive_features) is False

    anon_df = evaluator.anonymize_columns(X, sensitive_features=[sensitive_features[0]])

    assert evaluator.is_anonymous(anon_df, sensitive_features) is True
    assert len(anon_df.drop_duplicates()) > 1


def test_l_diversity_anonymize_full() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    sensitive_features = list(X.columns)[:2]
    kevaluator = kAnonymization()
    k_threshold = kevaluator.evaluate_data(
        GenericDataLoader(X, sensitive_features=sensitive_features)
    )

    levaluator = lDiversityDistinct()
    l_diversity = levaluator.evaluate_data(
        GenericDataLoader(X, sensitive_features=sensitive_features)
    )

    evaluator = DatasetAnonymization(
        k_threshold=k_threshold + 1, l_diversity=l_diversity + 1
    )
    assert evaluator.is_anonymous(X, sensitive_features) is False

    anon_df = evaluator.anonymize(X)

    assert evaluator.is_anonymous(anon_df, sensitive_features) is True


def test_t_closeness_sanity() -> None:
    evaluator = DatasetAnonymization(
        k_threshold=44,
        l_diversity=33,
        t_threshold=0.6,
        categorical_limit=11,
        max_partitions=22,
    )

    assert evaluator.categorical_limit == 11
    assert evaluator.k_threshold == 44
    assert evaluator.t_threshold == 0.6
    assert evaluator.l_diversity == 33
    assert evaluator.max_partitions == 22


def test_t_closeness_get_spans() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    evaluator = DatasetAnonymization(t_threshold=0.5)

    spans = evaluator._get_spans(X, X.index)

    assert set(spans.keys()) == set(X.columns)
    assert evaluator._get_categoricals(X) == ["sex"]


def test_t_closeness_split() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    evaluator = DatasetAnonymization(t_threshold=0.5)

    categoricals = evaluator._get_categoricals(X)
    for col in X.columns:
        lhs, rhs = evaluator._split(X, X.index, col, categoricals)

        assert len(lhs) > 0
        assert len(rhs) > 0
        assert len(lhs) + len(rhs) == len(X)

        assert evaluator._is_partition_anonymous(X, lhs, X.columns[0]) is True
        assert evaluator._is_partition_anonymous(X, rhs, X.columns[0]) is True


def test_t_closeness_partition() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    evaluator = DatasetAnonymization(t_threshold=0.1)
    partitions = evaluator._partition_dataset(X, list(X.columns), "sex")
    less_parts = len(partitions)

    evaluator = DatasetAnonymization(t_threshold=0.9)
    partitions = evaluator._partition_dataset(X, list(X.columns), "sex")
    more_parts = len(partitions)

    assert less_parts > 0
    assert more_parts > 0
    assert less_parts < more_parts


def test_t_closeness_validation() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    kevaluator = kAnonymization()
    k_threshold = kevaluator.evaluate_data(
        GenericDataLoader(X, sensitive_features=["sex"])
    )

    evaluator = DatasetAnonymization(k_threshold=1, t_threshold=0.2)
    assert evaluator.is_anonymous(X, sensitive_features=["sex"]) is True

    evaluator = DatasetAnonymization(k_threshold=k_threshold + 1, t_threshold=0.5)
    assert evaluator.is_anonymous(X, sensitive_features=["sex"]) is False


@pytest.mark.parametrize("t_threshold", [0.2, 0.5])
def test_t_closeness_anonymize_column(t_threshold: float) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    sensitive_features = ["target"]
    kevaluator = kAnonymization()
    k_threshold = kevaluator.evaluate_data(
        GenericDataLoader(X, sensitive_features=sensitive_features)
    )

    levaluator = lDiversityDistinct()
    l_diversity = levaluator.evaluate_data(
        GenericDataLoader(X, sensitive_features=sensitive_features)
    )

    evaluator = DatasetAnonymization(
        k_threshold=k_threshold + 1,
        l_diversity=l_diversity + 1,
        t_threshold=t_threshold,
    )
    assert evaluator.is_anonymous(X, sensitive_features) is False

    anon_df = evaluator.anonymize_column(X, sensitive_column=sensitive_features[0])

    assert evaluator.is_anonymous(anon_df, sensitive_features) is True
    assert len(anon_df.drop_duplicates()) > 1


@pytest.mark.parametrize("t_threshold", [0.2, 0.5])
def test_t_closeness_anonymize_columns(t_threshold: float) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    sensitive_features = list(X.columns)[:2]
    kevaluator = kAnonymization()
    k_threshold = kevaluator.evaluate_data(
        GenericDataLoader(X, sensitive_features=sensitive_features)
    )

    levaluator = lDiversityDistinct()
    l_diversity = levaluator.evaluate_data(
        GenericDataLoader(X, sensitive_features=sensitive_features)
    )

    evaluator = DatasetAnonymization(
        k_threshold=k_threshold + 1,
        l_diversity=l_diversity + 1,
        t_threshold=t_threshold,
    )
    assert evaluator.is_anonymous(X, sensitive_features) is False

    anon_df = evaluator.anonymize_column(X, sensitive_column=sensitive_features[0])

    assert evaluator.is_anonymous(anon_df, sensitive_features) is True
    assert len(anon_df.drop_duplicates()) > 1


@pytest.mark.parametrize("t_threshold", [0.2, 0.5])
def test_t_closeness_anonymize_full(t_threshold: float) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    sensitive_features = list(X.columns)[:2]
    kevaluator = kAnonymization()
    k_threshold = kevaluator.evaluate_data(
        GenericDataLoader(X, sensitive_features=sensitive_features)
    )

    levaluator = lDiversityDistinct()
    l_diversity = levaluator.evaluate_data(
        GenericDataLoader(X, sensitive_features=sensitive_features)
    )

    evaluator = DatasetAnonymization(
        k_threshold=k_threshold + 1,
        l_diversity=l_diversity + 1,
        t_threshold=t_threshold,
    )
    assert evaluator.is_anonymous(X, sensitive_features) is False

    anon_df = evaluator.anonymize(X)

    assert evaluator.is_anonymous(anon_df, sensitive_features) is True
