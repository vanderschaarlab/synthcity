# stdlib
from typing import Any, List

# third party
import pytest

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin


class AbstractMockPlugin(Plugin):
    pass


class MockPlugin(Plugin):
    @staticmethod
    def name() -> str:
        return "mock"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "Plugin":
        return self

    def _generate(self, *args: Any, **kwargs: Any) -> DataLoader:
        return GenericDataLoader([0])


def test_mock_plugin_fail() -> None:
    with pytest.raises(TypeError):
        AbstractMockPlugin()  # type: ignore


def test_mock_plugin_ok() -> None:
    plugin = MockPlugin()

    assert plugin.name() == "mock"
    assert plugin.type() == "debug"
    assert plugin.fit(GenericDataLoader([1])) == plugin
    assert plugin.generate().values == [1]

    buff = plugin.save()
    reloaded = Plugin.load(buff)

    assert reloaded.name() == plugin.name()
