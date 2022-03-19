# stdlib
from typing import Any, List

# third party
import pandas as pd
import pytest

# synthcity absolute
from synthcity.plugins.core.params import Params
from synthcity.plugins.core.plugin import Plugin


class AbstractMockPlugin(Plugin):
    pass


class MockPlugin(Plugin):
    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:
        return []

    @staticmethod
    def name() -> str:
        return "mock"

    @staticmethod
    def type() -> str:
        return "debug"

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "Plugin":
        return self

    def _generate(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return [1]


def test_mock_plugin_fail() -> None:
    with pytest.raises(TypeError):
        AbstractMockPlugin()  # type: ignore


def test_mock_plugin_ok() -> None:
    plugin = MockPlugin()

    assert plugin.name() == "mock"
    assert plugin.type() == "debug"
    assert plugin.fit(pd.DataFrame([])) == plugin
    assert plugin.generate().values == [1]
