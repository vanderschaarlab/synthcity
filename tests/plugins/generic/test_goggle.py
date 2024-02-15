# third party
import numpy as np
import pytest
from generic_helpers import generate_fixtures
from importlib_metadata import PackageNotFoundError, distribution
from sklearn.datasets import load_diabetes, load_iris

# synthcity absolute
from synthcity.metrics.eval import AlphaPrecision
from synthcity.plugins import Plugin
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins.generic.plugin_goggle import plugin
from synthcity.utils.serialization import load, save

is_missing_goggle_deps = plugin is None

plugin_name = "goggle"
plugin_args = {
    "n_iter": 10,
    "device": "cpu",
    "sampling_patience": 50,
}

if not is_missing_goggle_deps:
    goggle_dependencies = {"dgl", "torch-scatter", "torch-sparse", "torch-geometric"}
    missing_deps = []
    for dep in goggle_dependencies:
        try:
            distribution(dep)
        except PackageNotFoundError:
            missing_deps.append(dep)
    is_missing_goggle_deps = len(missing_deps) > 0


@pytest.mark.skipif(is_missing_goggle_deps, reason="Goggle dependencies not installed")
@pytest.mark.parametrize(
    "test_plugin",
    generate_fixtures(plugin_name, plugin),
)
def test_plugin_sanity(test_plugin: Plugin) -> None:
    assert test_plugin is not None


@pytest.mark.skipif(is_missing_goggle_deps, reason="Goggle dependencies not installed")
@pytest.mark.parametrize(
    "test_plugin",
    generate_fixtures(plugin_name, plugin),
)
def test_plugin_name(test_plugin: Plugin) -> None:
    assert test_plugin.name() == plugin_name


@pytest.mark.skipif(is_missing_goggle_deps, reason="Goggle dependencies not installed")
@pytest.mark.parametrize(
    "test_plugin",
    generate_fixtures(plugin_name, plugin),
)
def test_plugin_type(test_plugin: Plugin) -> None:
    assert test_plugin.type() == "generic"


@pytest.mark.skipif(is_missing_goggle_deps, reason="Goggle dependencies not installed")
@pytest.mark.parametrize(
    "test_plugin",
    generate_fixtures(plugin_name, plugin),
)
def test_plugin_hyperparams(test_plugin: Plugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 9


@pytest.mark.skipif(is_missing_goggle_deps, reason="Goggle dependencies not installed")
@pytest.mark.parametrize(
    "test_plugin",
    generate_fixtures(plugin_name, plugin, plugin_args),
)
def test_plugin_fit(test_plugin: Plugin) -> None:
    Xraw, y = load_diabetes(return_X_y=True, as_frame=True)
    Xraw["target"] = y
    test_plugin.fit(GenericDataLoader(Xraw))


@pytest.mark.skipif(is_missing_goggle_deps, reason="Goggle dependencies not installed")
@pytest.mark.parametrize(
    "test_plugin",
    generate_fixtures(plugin_name, plugin, plugin_args),
)
@pytest.mark.parametrize("serialize", [True, False])
def test_plugin_generate(test_plugin: Plugin, serialize: bool) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y
    test_plugin.fit(GenericDataLoader(X))

    if serialize:
        saved = save(test_plugin)
        test_plugin = load(saved)

    X_gen = test_plugin.generate()
    assert len(X_gen) == len(X)
    assert X_gen.shape[1] == X.shape[1]
    assert test_plugin.schema_includes(X_gen)

    X_gen = test_plugin.generate(50)
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)

    # generate with random seed
    X_gen1 = test_plugin.generate(50, random_state=0)
    X_gen2 = test_plugin.generate(50, random_state=0)
    X_gen3 = test_plugin.generate(50)
    assert (X_gen1.numpy() == X_gen2.numpy()).all()
    assert (X_gen1.numpy() != X_gen3.numpy()).any()


@pytest.mark.skipif(is_missing_goggle_deps, reason="Goggle dependencies not installed")
@pytest.mark.parametrize(
    "test_plugin",
    generate_fixtures(plugin_name, plugin, plugin_args),
)
def test_plugin_generate_constraints_goggle(test_plugin: Plugin) -> None:
    X, y = load_iris(as_frame=True, return_X_y=True)
    X["target"] = y
    test_plugin.fit(GenericDataLoader(X))

    constraints = Constraints(
        rules=[
            ("target", "eq", 1),
        ]
    )

    X_gen = test_plugin.generate(constraints=constraints).dataframe()
    assert len(X_gen) == len(X)
    assert test_plugin.schema_includes(X_gen)
    assert constraints.filter(X_gen).sum() == len(X_gen)
    assert (X_gen["target"] == 1).all()

    X_gen = test_plugin.generate(count=50, constraints=constraints).dataframe()
    assert len(X_gen) == 50
    assert test_plugin.schema_includes(X_gen)
    assert constraints.filter(X_gen).sum() == len(X_gen)
    assert list(X_gen.columns) == list(X.columns)


@pytest.mark.skipif(is_missing_goggle_deps, reason="Goggle dependencies not installed")
def test_sample_hyperparams() -> None:
    assert plugin is not None
    for i in range(100):
        args = plugin.sample_hyperparameters()
        assert plugin(**args) is not None


# TODO: Known issue goggle seems to have a performance issue.
# Testing fidelity instead. Also need to test more architectures
@pytest.mark.skipif(is_missing_goggle_deps, reason="Goggle dependencies not installed")
@pytest.mark.slow_2
@pytest.mark.slow
def test_eval_fidelity_goggle(compress_dataset: bool, decoder_arch: str) -> None:
    results = []
    Xraw, y = load_iris(return_X_y=True, as_frame=True)
    Xraw["target"] = y
    X = GenericDataLoader(Xraw)

    assert plugin is not None
    for retry in range(3):
        test_plugin = plugin(
            encoder_dim=32,
            encoder_l=4,
            decoder_dim=32,
            decoder_l=4,
            data_encoder_max_clusters=20,
            compress_dataset=False,
            decoder_arch="gcn",
            random_state=retry,
        )
        evaluator = AlphaPrecision()

        test_plugin.fit(X)
        X_syn = test_plugin.generate(count=len(X), random_state=retry)
        eval_results = evaluator.evaluate(X, X_syn)
        results.append(eval_results["authenticity_OC"])

    assert np.mean(results) > 0.7
