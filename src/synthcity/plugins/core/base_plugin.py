# stdlib
from abc import ABCMeta, abstractmethod
from importlib.abc import Loader
import importlib.util
from os.path import basename
from pathlib import Path
from typing import Any, Dict, Generator, List, Type

# third party
import pandas as pd

# synthcity absolute
import synthcity.logger as log
import synthcity.plugins.core.cast as cast

# synthcity relative
from .params import Params


class Plugin(metaclass=ABCMeta):
    """Base class for all plugins.
    Each derived class must implement the following methods:
        hyperparameter_space() - a static method that returns the hyperparameters that can be tuned during AutoML.
        type() - a static method that returns the type of the plugin. e.g., imputation, preprocessing, prediction, etc.
        name() - a static method that returns the name of the plugin. e.g., EM, mice, etc.
        _fit() - internal method, called by `fit` on each training set.
        _generate() - internal method, called by `generate`.

    If any method implementation is missing, the class constructor will fail.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    @abstractmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:
        ...

    @classmethod
    def sample_hyperparameters(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        param_space = cls.hyperparameter_space(*args, **kwargs)

        results = {}

        for hp in param_space:
            results[hp.name] = hp.sample()

        return results

    @staticmethod
    @abstractmethod
    def name() -> str:
        ...

    @staticmethod
    @abstractmethod
    def type() -> str:
        ...

    @classmethod
    def fqdn(cls) -> str:
        return cls.type() + "." + cls.name()

    def fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "Plugin":
        X = cast.to_dataframe(X)
        return self._fit(X, *args, **kwargs)

    @abstractmethod
    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "Plugin":
        ...

    def generate(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame(self._generate(*args, **kwargs))

    @abstractmethod
    def _generate(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        ...


class PluginLoader:
    def __init__(self, plugins: list, expected_type: Type) -> None:
        self._plugins: Dict[str, Type] = {}
        self._available_plugins = {}
        for plugin in plugins:
            stem = Path(plugin).stem.split("plugin_")[-1]
            self._available_plugins[stem] = plugin

        self._expected_type = expected_type

    def _load_single_plugin(self, plugin: str) -> None:
        name = basename(plugin)
        failed = False
        for retry in range(2):
            try:
                spec = importlib.util.spec_from_file_location(name, plugin)
                if not isinstance(spec.loader, Loader):
                    raise RuntimeError("invalid plugin type")

                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                cls = mod.plugin  # type: ignore
                failed = False
                break
            except BaseException as e:
                log.critical(f"load failed: {e}")
                failed = True
                break

        if failed:
            log.critical(f"module {name} load failed")
            return

        log.debug(f"Loaded plugin {cls.type()} - {cls.name()}")
        self.add(cls.name(), cls)

    def list(self) -> List[str]:
        all_plugins = list(self._plugins.keys()) + list(self._available_plugins.keys())
        return list(set(all_plugins))

    def types(self) -> List[Type]:
        return list(self._plugins.values())

    def add(self, name: str, cls: Type) -> "PluginLoader":
        if name in self._plugins:
            raise ValueError(f"Plugin {name} already exists.")

        if not issubclass(cls, self._expected_type):
            raise ValueError(
                f"Plugin {name} must derive the {self._expected_type} interface."
            )

        self._plugins[name] = cls

        return self

    def get(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name not in self._plugins and name not in self._available_plugins:
            raise ValueError(f"Plugin {name} doesn't exist.")

        if name not in self._plugins:
            self._load_single_plugin(self._available_plugins[name])

        if name not in self._plugins:
            raise ValueError(f"Plugin {name} cannot be loaded.")

        return self._plugins[name](*args, **kwargs)

    def get_type(self, name: str) -> Type:
        if name not in self._plugins and name not in self._available_plugins:
            raise ValueError(f"Plugin {name} doesn't exist.")

        if name not in self._plugins:
            self._load_single_plugin(self._available_plugins[name])

        if name not in self._plugins:
            raise ValueError(f"Plugin {name} doesn't exist.")

        return self._plugins[name]

    def __iter__(self) -> Generator:
        for x in self._plugins:
            yield x

    def __len__(self) -> int:
        return len(self.list())

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def reload(self) -> "PluginLoader":
        self._plugins = {}
        return self
