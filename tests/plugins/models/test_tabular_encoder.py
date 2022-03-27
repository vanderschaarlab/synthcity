# third party
import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_digits

# synthcity absolute
from synthcity.plugins.models import TabularEncoder


def test_encoder_sanity() -> None:
    X, _ = load_digits(return_X_y=True, as_frame=True)
    net = TabularEncoder(max_clusters=4, weight_threshold=0.1)

    assert net._max_clusters == 4
    assert net._weight_threshold == 0.1


@pytest.mark.parametrize("max_clusters", [4, 10])
def test_encoder_fit_no_discrete_param(max_clusters: int) -> None:
    X, _ = load_diabetes(return_X_y=True, as_frame=True)
    net = TabularEncoder(max_clusters=max_clusters, weight_threshold=0.1)

    net.fit(X)

    layout = net.layout()

    for column in layout:
        assert column.column_type in ["discrete", "continuous"]
        if column.column_type == "discrete":
            assert len(X[column.column_name].unique()) < 20
            assert column.output_dimensions == len(X[column.column_name].unique())
        else:
            assert len(X[column.column_name].unique()) > 10
            assert column.output_dimensions <= 1 + max_clusters


@pytest.mark.parametrize("max_clusters", [4, 10])
def test_encoder_fit_discrete_param(max_clusters: int) -> None:
    X, _ = load_diabetes(return_X_y=True, as_frame=True)
    net = TabularEncoder(max_clusters=max_clusters, weight_threshold=0.1)

    net.fit(X, discrete_columns=["sex"])

    layout = net.layout()

    for column in layout:
        assert column.column_type in ["discrete", "continuous"]
        if column.column_type == "discrete":
            assert len(X[column.column_name].unique()) < 20
            assert column.output_dimensions == len(X[column.column_name].unique())
        else:
            assert len(X[column.column_name].unique()) > 10
            assert column.output_dimensions <= 1 + max_clusters


@pytest.mark.parametrize("max_clusters", [4, 10])
def test_encoder_fit_transform(max_clusters: int) -> None:
    X, _ = load_diabetes(return_X_y=True, as_frame=True)
    net = TabularEncoder(max_clusters=max_clusters, weight_threshold=0.1)

    net.fit(X)
    layout = net.layout()

    encoded = net.transform(X)

    assert (X.index == encoded.index).all()

    for column in layout:
        if column.column_type == "discrete":
            for val in X[column.column_name].unique():
                assert f"{column.column_name}_{val}" in encoded.columns
                assert set(encoded[f"{column.column_name}_{val}"].unique()) == set(
                    [0, 1]
                )

        else:
            assert f"{column.column_name}.normalized" in encoded.columns
            for enc_col in encoded.columns:
                if column.column_name in enc_col and "normalized" not in enc_col:
                    assert set(encoded[enc_col].unique()) == set([0, 1])


@pytest.mark.parametrize("max_clusters", [20, 50])
def test_encoder_inverse_transform(max_clusters: int) -> None:
    X, _ = load_diabetes(return_X_y=True, as_frame=True)
    net = TabularEncoder(max_clusters=max_clusters, weight_threshold=0.1)

    net.fit(X)

    encoded = net.transform(X)
    recovered = net.inverse_transform(encoded)

    assert X.shape == recovered.shape
    assert (X.columns == recovered.columns).all()
    assert np.abs(X - recovered).sum().sum() < 5


def test_encoder_activation_layout() -> None:
    X, _ = load_diabetes(return_X_y=True, as_frame=True)
    net = TabularEncoder()

    net.fit(X)
    act_layout = net.activation_layout(
        discrete_activation="softmax", continuous_activation="tanh"
    )
    layout = net.layout()

    for col_act, col_layout in zip(act_layout, layout):
        act, size = col_act
        col_info = col_layout

        if act == "softmax":
            assert col_info.column_type == "discrete"
        else:
            assert col_info.column_type == "continuous"

        assert size == col_info.output_dimensions
