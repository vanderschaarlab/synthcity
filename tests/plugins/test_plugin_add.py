# stdlib
from typing import Any, List

# third party
from sklearn.datasets import load_breast_cancer

# synthcity absolute
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class DummyCopyDataPlugin(Plugin):
    """Dummy plugin for debugging."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "copy_data"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "DummyCopyDataPlugin":
        self.features_count = X.shape[1]
        self.X = X
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> DataLoader:
        return self.X.sample(count)


def test_add_dummy_plugin() -> None:
    generators = Plugins()
    assert "copy_data" not in generators.list()
    # Add the new plugin to the collection
    generators.add("copy_data", DummyCopyDataPlugin)

    # Load reference data
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    loader = GenericDataLoader(X)
    loader.dataframe()

    # Train the new plugin
    gen = generators.get("copy_data")
    gen.fit(loader)

    # Generate some new data
    gen.generate(count=10)

    # Test that the new plugin is in the list of available plugins
    available_plugins = Plugins().list()
    print(available_plugins)
    assert "copy_data" in available_plugins
