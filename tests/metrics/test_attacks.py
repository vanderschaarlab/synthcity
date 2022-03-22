# third party
from sklearn.datasets import load_diabetes

# synthcity absolute
from synthcity.metrics.attacks import evaluate_sensitive_data_leakage
from synthcity.plugins import Plugins


def test_evaluate_sensitive_data_leakage() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    # Sampler
    test_plugin = Plugins().get("dummy_sampler")
    test_plugin.fit(X)
    X_gen = test_plugin.generate(2 * len(X))

    score = evaluate_sensitive_data_leakage(
        X.drop(columns=["target"]),
        X["target"],
        X_gen.drop(columns=["target"]),
        X_gen["target"],
    )
    assert score == 0

    score = evaluate_sensitive_data_leakage(
        X.drop(columns=["target"]),
        X["target"],
        X_gen.drop(columns=["target"]),
        X_gen["target"],
        sensitive_columns=["sex"],
    )
    assert score > 0.5

    score = evaluate_sensitive_data_leakage(
        X.drop(columns=["target"]),
        X["target"],
        X_gen.drop(columns=["target"]),
        X_gen["target"],
        sensitive_columns=["age"],
    )
    assert score < 1

    # Random noise

    test_plugin = Plugins().get("random_noise")
    test_plugin.fit(X)
    X_gen = test_plugin.generate(2 * len(X))

    score = evaluate_sensitive_data_leakage(
        X.drop(columns=["target"]),
        X["target"],
        X_gen.drop(columns=["target"]),
        X_gen["target"],
        sensitive_columns=["sex"],
    )
    assert score < 1
