# third party
import pytest
from sklearn.datasets import load_breast_cancer

# synthcity absolute
from synthcity.metrics.anonymity import kAnonimity


def test_k_anonymity_sanity() -> None:
    evaluator = kAnonimity()

    assert evaluator.categorical_limit == 5


def test_k_anonymity_get_spans() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    evaluator = kAnonimity()

    spans = evaluator._get_spans(X, X.index)

    assert set(spans.keys()) == set(X.columns)


def test_k_anonymity_split() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    evaluator = kAnonimity()

    categoricals = evaluator._get_categoricals(X)
    for col in X.columns:
        lhs, rhs = evaluator._split(X, X.index, col, categoricals)

        assert len(lhs) > 0
        assert len(rhs) > 0
        assert len(lhs) + len(rhs) == len(X)

        assert evaluator._is_index_k_anonymous(lhs) is True
        assert evaluator._is_index_k_anonymous(rhs) is True


def test_k_anonymity_partition() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    evaluator = kAnonimity(k=100)
    partitions = evaluator._partition_dataset(X, list(X.columns))
    less_parts = len(partitions)

    evaluator = kAnonimity(k=5)
    partitions = evaluator._partition_dataset(X, list(X.columns))
    more_parts = len(partitions)

    assert less_parts < more_parts


def test_k_anonymity_validation() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    evaluator = kAnonimity(k=1)
    assert evaluator.is_k_anonymous(X) is True

    evaluator = kAnonimity(k=3)
    assert evaluator.is_k_anonymous(X) is False


@pytest.mark.parametrize("k", [2, 5, 15, 30])
def test_k_anonymity_anonymize(k: int) -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X["target"] = y

    evaluator = kAnonimity(k=k)
    anon_df = evaluator.anonymize(X, sensitive_column="target")

    assert evaluator.is_k_anonymous(anon_df, "target") is True
