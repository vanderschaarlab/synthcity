# stdlib
import importlib.util
from abc import ABCMeta, abstractmethod
from importlib.abc import Loader
from os.path import basename
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Type, Union

# third party
import pandas as pd
from pydantic import validate_arguments

# synthcity absolute
import synthcity.logger as log
from synthcity.metrics.plots import (
    plot_associations_comparison,
    plot_marginal_comparison,
    plot_tsne,
)
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import (
    DataLoader,
    GenericDataLoader,
    TimeSeriesDataLoader,
    create_from_info,
)
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE


class Plugin(metaclass=ABCMeta):
    """Base class for all plugins.
    Each derived class must implement the following methods:
        type() - a static method that returns the type of the plugin. e.g., debug, generative, bayesian, etc.
        name() - a static method that returns the name of the plugin. e.g., ctgan, random_noise, etc.
        hyperparameter_space() - a static method that returns the hyperparameters that can be tuned during AutoML.
        _fit() - internal method, called by `fit` on each training set.
        _generate() - internal method, called by `generate`.

    If any method implementation is missing, the class constructor will fail.

    Constructor Args:
        strict: float.
            If True, is raises an exception if the generated data is not following the requested constraints. If False, it returns only the rows that match the constraints.
    """

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    def __init__(
        self,
        sampling_strategy: str = "marginal",  # uniform, marginal
        sampling_patience: int = 500,
        strict: bool = True,
        device: Any = DEVICE,
    ) -> None:
        """

        Args:
            sampling_strategy: str
                Internal sampling strategy [marginal, uniform].
            strict: bool
                If True, the generation process will raise an exception if the synthetic data doesn't satisfy the generation constraints. If False, the generation process will return only the valid rows under the constraint.

        """
        self._schema: Optional[Schema] = None
        self.sampling_strategy = sampling_strategy
        self.sampling_patience = sampling_patience
        self.strict = strict
        self.device = device

    @staticmethod
    @abstractmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        """Returns the hyperparameter space for the derived plugin."""
        ...

    @classmethod
    def sample_hyperparameters(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Sample value from the hyperparameter space for the current plugin."""
        param_space = cls.hyperparameter_space(*args, **kwargs)

        results = {}

        for hp in param_space:
            results[hp.name] = hp.sample()[0]

        return results

    @staticmethod
    @abstractmethod
    def name() -> str:
        """The name of the plugin."""
        ...

    @staticmethod
    @abstractmethod
    def type() -> str:
        """The type of the plugin."""
        ...

    @classmethod
    def fqdn(cls) -> str:
        """The Fully-Qualified name of the plugin."""
        return cls.type() + "." + cls.name()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: Union[DataLoader, pd.DataFrame], *args: Any, **kwargs: Any) -> Any:
        """Training method the synthetic data plugin.

        Args:
            X: DataLoader.
                The reference dataset.

        Returns:
            self
        """
        if isinstance(X, (pd.DataFrame)):
            X = GenericDataLoader(X)

        self.data_info = X.info()
        self._schema = Schema(
            data=X,
            sampling_strategy=self.sampling_strategy,
        )

        return self._fit(X, *args, **kwargs)

    @abstractmethod
    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "Plugin":
        """Internal training method the synthetic data plugin.

        Args:
            X: DataLoader.
                The reference dataset.

        Returns:
            self
        """
        ...

    @validate_arguments
    def generate(
        self,
        count: Optional[int] = None,
        constraints: Optional[Constraints] = None,
        **kwargs: Any,
    ) -> DataLoader:
        """Synthetic data generation method.

        Args:
            count: optional int.
                The number of samples to generate. If None, it generated len(reference_dataset) samples.
            constraints: optional Constraints
                Optional constraints to apply on the generated data. If none, the reference schema constraints are applied.

        Returns:
            <count> synthetic samples
        """
        if self._schema is None:
            raise RuntimeError("Fit the model first")

        if count is None:
            count = self.data_info["len"]

        gen_constraints = self.schema().as_constraints()
        if constraints is not None:
            gen_constraints = gen_constraints.extend(constraints)

        syn_schema = Schema.from_constraints(gen_constraints)

        X_syn = self._generate(count=count, syn_schema=syn_schema, **kwargs)

        if not X_syn.satisfies(gen_constraints) and self.strict:
            raise RuntimeError(
                f"Plugin {self.name()} failed to meet the synthetic constraints."
            )

        return X_syn.match(gen_constraints)

    @abstractmethod
    def _generate(
        self,
        count: int,
        syn_schema: Schema,
        **kwargs: Any,
    ) -> DataLoader:
        """Internal synthetic data generation method.

        Args:
            count: optional int.
                The number of samples to generate. If None, it generated len(reference_dataset) samples.
            syn_schema:
                The schema/constraints that need to be satisfied by the synthetic data.

        Returns:
            <count> synthetic samples
        """
        ...

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _safe_generate(
        self, gen_cbk: Callable, count: int, syn_schema: Schema, **kwargs: Any
    ) -> DataLoader:
        constraints = syn_schema.as_constraints()

        data_synth = pd.DataFrame([], columns=self.schema().features())
        for it in range(self.sampling_patience):
            # sample
            iter_samples = gen_cbk(count, **kwargs)
            iter_samples_df = pd.DataFrame(
                iter_samples, columns=self.schema().features()
            )
            # validate schema
            iter_samples_df = self.schema().adapt_dtypes(iter_samples_df)

            iter_synth_valid = constraints.match(iter_samples_df)
            data_synth = pd.concat([data_synth, iter_synth_valid], ignore_index=True)

            if len(data_synth) >= count:
                break

        data_synth = self.schema().adapt_dtypes(data_synth).head(count)

        return create_from_info(data_synth, self.data_info)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _safe_generate_time_series(
        self, gen_cbk: Callable, count: int, syn_schema: Schema, **kwargs: Any
    ) -> DataLoader:
        assert self.data_info["data_type"] == "time_series"

        constraints = syn_schema.as_constraints()

        data_synth = pd.DataFrame([], columns=self.schema().features())
        for it in range(self.sampling_patience):
            # sample
            static, temporal, outcome = gen_cbk(count, **kwargs)
            loader = TimeSeriesDataLoader(
                temporal_data=temporal, static_data=static, outcome=outcome
            )
            iter_samples_df = loader.dataframe()

            # validate schema
            iter_samples_df = self.schema().adapt_dtypes(iter_samples_df)

            iter_synth_valid = constraints.match(iter_samples_df)
            data_synth = pd.concat([data_synth, iter_synth_valid], ignore_index=True)

            if len(data_synth) >= count:
                break

        data_synth = self.schema().adapt_dtypes(data_synth).head(count)

        return create_from_info(data_synth, self.data_info)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def schema_includes(self, other: Union[DataLoader, pd.DataFrame]) -> bool:
        """Helper method to test if the reference schema includes a Dataset

        Args:
            other: DataLoader.
                The dataset to test

        Returns:
            bool, if the schema includes the dataset or not.

        """
        other_schema = Schema(data=other)
        return self.schema().includes(other_schema)

    def schema(self) -> Schema:
        """The reference schema"""
        if self._schema is None:
            raise RuntimeError("Fit the model first")

        return self._schema

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def plot(self, plt: Any, X: DataLoader, *args: Any, **kwargs: Any) -> Any:
        """Plot the real-synthetic distributions.

        Args:
            plt: output
            X: DataLoader.
                The reference dataset.

        Returns:
            self
        """
        X_syn = self.generate()

        plot_marginal_comparison(plt, X, X_syn)
        plot_associations_comparison(plt, X, X_syn)
        plot_tsne(plt, X, X_syn)


class PluginLoader:
    """Plugin loading utility class.
    Used to load the plugins from the current folder.
    """

    @validate_arguments
    def __init__(self, plugins: list, expected_type: Type) -> None:
        self._plugins: Dict[str, Type] = {}
        self._available_plugins = {}
        for plugin in plugins:
            stem = Path(plugin).stem.split("plugin_")[-1]
            self._available_plugins[stem] = plugin

        self._expected_type = expected_type

    @validate_arguments
    def _load_single_plugin(self, plugin: str) -> None:
        """Helper for loading a single plugin"""
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

    def list(self, category: str = "generic") -> List[str]:
        """Get all the available plugins."""
        all_plugins = list(self._plugins.keys()) + list(self._available_plugins.keys())
        plugins = []
        for plugin in all_plugins:
            if self.get_type(plugin).type() == category:
                plugins.append(plugin)

        return list(set(plugins))

    def types(self) -> List[Type]:
        """Get the loaded plugins types"""
        return list(self._plugins.values())

    def add(self, name: str, cls: Type) -> "PluginLoader":
        """Add a new plugin"""
        if name in self._plugins:
            log.info(f"Plugin {name} already exists. Overwriting")

        if not issubclass(cls, self._expected_type):
            raise ValueError(
                f"Plugin {name} must derive the {self._expected_type} interface."
            )

        self._plugins[name] = cls

        return self

    @validate_arguments
    def get(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Create a new object from a plugin.
        Args:
            name: str. The name of the plugin
            &args, **kwargs. Plugin specific arguments

        Returns:
            The new object
        """
        if name not in self._plugins and name not in self._available_plugins:
            raise ValueError(f"Plugin {name} doesn't exist.")

        if name not in self._plugins:
            self._load_single_plugin(self._available_plugins[name])

        if name not in self._plugins:
            raise ValueError(f"Plugin {name} cannot be loaded.")

        return self._plugins[name](*args, **kwargs)

    @validate_arguments
    def get_type(self, name: str) -> Type:
        """Get the class type of a plugin.
        Args:
            name: str. The name of the plugin

        Returns:
            The class of the plugin
        """
        if name not in self._plugins and name not in self._available_plugins:
            raise ValueError(f"Plugin {name} doesn't exist.")

        if name not in self._plugins:
            self._load_single_plugin(self._available_plugins[name])

        if name not in self._plugins:
            raise ValueError(f"Plugin {name} doesn't exist.")

        return self._plugins[name]

    def __iter__(self) -> Generator:
        """Iterate the loaded plugins."""
        for x in self._plugins:
            yield x

    def __len__(self) -> int:
        """The number of available plugins."""
        return len(self.list())

    @validate_arguments
    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def reload(self) -> "PluginLoader":
        self._plugins = {}
        return self
