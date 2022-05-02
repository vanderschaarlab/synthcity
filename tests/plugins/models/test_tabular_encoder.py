# third party
import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_digits

# synthcity absolute
from synthcity.plugins.models import TabularEncoder


def test_encoder_sanity() -> None:
    X, _ = load_digits(return_X_y=True, as_frame=True)
    net = TabularEncoder(max_clusters=4, weight_threshold=0.1)

    assert net.max_clusters == 4
    assert net.weight_threshold == 0.1


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

    encoded = net.fit_transform(X)
    layout = net.layout()

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

    assert len(layout) <= len(act_layout)

    act_step = 0

    for col_info in layout:
        if col_info.column_type == "continuous":
            assert act_layout[act_step] == ("tanh", 1)
            assert act_layout[act_step + 1] == (
                "softmax",
                col_info.output_dimensions - 1,
            )
            act_step += 2
        else:
            assert act_layout[act_step] == ("softmax", col_info.output_dimensions)
            act_step += 1
