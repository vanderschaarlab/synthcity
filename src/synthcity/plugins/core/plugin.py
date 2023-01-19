# stdlib
import importlib.util
import platform
import sys
from abc import ABCMeta, abstractmethod
from importlib.abc import Loader
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
    TimeSeriesSurvivalDataLoader,
    create_from_info,
)
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.core.serializable import Serializable
from synthcity.utils.constants import DEVICE
from synthcity.utils.reproducibility import enable_reproducible_results
from synthcity.utils.serialization import load_from_file, save_to_file


class Plugin(Serializable, metaclass=ABCMeta):
    """
    .. inheritance-diagram:: synthcity.plugins.core.plugin.Plugin
        :parts: 1

    Base class for all plugins.

    Each derived class must implement the following methods:
        type() - a static method that returns the type of the plugin. e.g., debug, generative, bayesian, etc.
        name() - a static method that returns the name of the plugin. e.g., ctgan, random_noise, etc.
        hyperparameter_space() - a static method that returns the hyperparameters that can be tuned during AutoML.
        _fit() - internal method, called by `fit` on each training set.
        _generate() - internal method, called by `generate`.

    If any method implementation is missing, the class constructor will fail.

    Args:
        strict: bool. Default = True
            If True, is raises an exception if the generated data is not following the requested constraints. If False, it returns only the rows that match the constraints.
        workspace: Path
            Path for caching intermediary results
        compress_dataset: bool. Default = False
            Drop redundant features before training the generator.
        device:
            PyTorch device: cpu or cuda.
        random_state: int
            Random seed
        sampling_patience: int.
            Max inference iterations to wait for the generated data to match the training schema.
        sampling_strategy: str
            Internal parameter for schema. marginal or uniform.
    """

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    def __init__(
        self,
        sampling_patience: int = 500,
        strict: bool = True,
        device: Any = DEVICE,
        random_state: int = 0,
        workspace: Path = Path("workspace"),
        compress_dataset: bool = False,
        sampling_strategy: str = "marginal",  # uniform, marginal
    ) -> None:
        super().__init__()

        enable_reproducible_results(random_state)

        self._schema: Optional[Schema] = None
        self._training_schema: Optional[Schema] = None
        self._data_encoders: Optional[Dict] = None

        self.sampling_strategy = sampling_strategy
        self.sampling_patience = sampling_patience
        self.strict = strict
        self.device = device
        self.random_state = random_state
        self.compress_dataset = compress_dataset

        workspace.mkdir(parents=True, exist_ok=True)
        self.workspace = workspace

        self.fitted = False
        self.expecting_conditional = False

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

    @classmethod
    def sample_hyperparameters_optuna(
        cls, trial: Any, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        param_space = cls.hyperparameter_space(*args, **kwargs)

        results = {}

        for hp in param_space:
            if isinstance(hp, IntegerDistribution):
                results[hp.name] = trial.suggest_int(hp.name, hp.low, hp.high, hp.step)
            elif isinstance(hp, FloatDistribution):
                results[hp.name] = trial.suggest_float(hp.name, hp.low, hp.high)
            elif isinstance(hp, CategoricalDistribution):
                results[hp.name] = trial.suggest_categorical(hp.name, hp.choices)
            else:
                raise RuntimeError(f"unknown distribution type {hp}")

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
            cond: Optional, Union[pd.DataFrame, pd.Series, np.ndarray]
                Optional Training Conditional.
                The training conditional can be used to control to output of some models, like GANs or VAEs. The content can be anything, as long as it maps to the training dataset X.
                Usage example:
                    >>> from sklearn.datasets import load_iris
                    >>> from synthcity.plugins.core.dataloader import GenericDataLoader
                    >>> from synthcity.plugins.core.constraints import Constraints
                    >>>
                    >>> # Load in `test_plugin` the generative model of choice
                    >>> # ....
                    >>>
                    >>> X, y = load_iris(as_frame=True, return_X_y=True)
                    >>> X["target"] = y
                    >>>
                    >>> X = GenericDataLoader(X)
                    >>> test_plugin.fit(X, cond=y)
                    >>>
                    >>> count = 10
                    >>> X_gen = test_plugin.generate(count, cond=np.ones(count))
                    >>>
                    >>> # The Conditional only optimizes the output generation
                    >>> # for GANs and VAEs, but does NOT guarantee the samples
                    >>> # are only from that condition.
                    >>> # If you want to guarantee that output contains only
                    >>> # "target" == 1 samples, use Constraints.
                    >>>
                    >>> constraints = Constraints(
                    >>>     rules=[
                    >>>         ("target", "==", 1),
                    >>>     ]
                    >>> )
                    >>> X_gen = test_plugin.generate(count,
                    >>>         cond=np.ones(count),
                    >>>         constraints=constraints
                    >>>        )
                    >>> assert (X_gen["target"] == 1).all()

        Returns:
            self
        """
        if isinstance(X, (pd.DataFrame)):
            X = GenericDataLoader(X)

        if "cond" in kwargs and kwargs["cond"] is not None:
            self.expecting_conditional = True

        self.data_info = X.info()

        self._schema = Schema(
            data=X,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
        )

        X, self._data_encoders = X.encode()
        if self.compress_dataset:
            X_hash = X.hash()
            bkp_file = (
                self.workspace
                / f"compressed_df_{X_hash}_{platform.python_version()}.bkp"
            )
            if not bkp_file.exists():
                X_compressed_context = X.compress()
                save_to_file(bkp_file, X_compressed_context)

            X, self.compress_context = load_from_file(bkp_file)

        self._training_schema = Schema(
            data=X,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
        )

        output = self._fit(X, *args, **kwargs)
        self.fitted = True

        return output

    @abstractmethod
    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "Plugin":
        """Internal training method the synthetic data plugin.

        Args:
            X: DataLoader.
                The reference dataset.
            cond: Optional, Union[pd.DataFrame, pd.Series, np.ndarray]
                Training Conditional
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
            cond: Optional, Union[pd.DataFrame, pd.Series, np.ndarray].
                Optional Generation Conditional. The conditional can be used only if the model was trained using a conditional too.
                If provided, it must have `count` length.
                Not all models support conditionals. The conditionals can be used in VAEs or GANs to speed-up the generation under some constraints. For model agnostic solutions, check out the `constraints` parameter.
            constraints: optional Constraints.
                Optional constraints to apply on the generated data. If none, the reference schema constraints are applied. The constraints are model agnostic, and will filter the output of the generative model.
                The constraints are a list of rules. Each rule is a tuple of the form (<feature>, <operation>, <value>).

                Valid Operations:
                    - "<", "lt" : less than <value>
                    - "<=", "le": less or equal with <value>
                    - ">", "gt" : greater than <value>
                    - ">=", "ge": greater or equal with <value>
                    - "==", "eq": equal with <value>
                    - "in": valid for categorical features, and <value> must be array. for example, ("target", "in", [0, 1])
                    - "dtype": <value> can be a data type. For example, ("target", "dtype", "int")

                Usage example:
                    >>> from synthcity.plugins.core.constraints import Constraints
                    >>> constraints = Constraints(
                    >>>   rules=[
                    >>>             ("InterestingFeature", "==", 0),
                    >>>         ]
                    >>>     )
                    >>>
                    >>> syn_data = syn_model.generate(
                            count=count,
                            constraints=constraints
                        ).dataframe()
                    >>>
                    >>> assert (syn_data["InterestingFeature"] == 0).all()

        Returns:
            <count> synthetic samples
        """
        if not self.fitted:
            raise RuntimeError("Fit the generator first")

        if self._schema is None:
            raise RuntimeError("Fit the model first")

        has_gen_cond = "cond" in kwargs and kwargs["cond"] is not None
        if has_gen_cond and not self.expecting_conditional:
            raise RuntimeError(
                "Conditional mismatch. Got inference conditional, without any training conditional"
            )

        if count is None:
            count = self.data_info["len"]

        # We use the training schema for the generation
        gen_constraints = self.training_schema().as_constraints()
        if constraints is not None:
            gen_constraints = gen_constraints.extend(constraints)

        syn_schema = Schema.from_constraints(gen_constraints)

        X_syn = self._generate(count=count, syn_schema=syn_schema, **kwargs)
        if self.compress_dataset:
            X_syn = X_syn.decompress(self.compress_context)
        if self._data_encoders is not None:
            X_syn = X_syn.decode(self._data_encoders)

        # The dataset is decompressed here, we can use the public schema
        gen_constraints = self.schema().as_constraints()
        if constraints is not None:
            gen_constraints = gen_constraints.extend(constraints)

        if not X_syn.satisfies(gen_constraints) and self.strict:
            raise RuntimeError(
                f"Plugin {self.name()} failed to meet the synthetic constraints."
            )

        if self.strict:
            X_syn = X_syn.match(gen_constraints)

        return X_syn

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
            cond: Optional, Union[pd.DataFrame, pd.Series, np.ndarray]
                Generation Conditional

        Returns:
            <count> synthetic samples
        """
        ...

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _safe_generate(
        self, gen_cbk: Callable, count: int, syn_schema: Schema, **kwargs: Any
    ) -> DataLoader:
        constraints = syn_schema.as_constraints()

        data_synth = pd.DataFrame([], columns=self.training_schema().features())
        for it in range(self.sampling_patience):
            # sample
            iter_samples = gen_cbk(count, **kwargs)
            iter_samples_df = pd.DataFrame(
                iter_samples, columns=self.training_schema().features()
            )

            # validate schema
            iter_samples_df = self.training_schema().adapt_dtypes(iter_samples_df)

            if self.strict:
                iter_samples_df = constraints.match(iter_samples_df)
                iter_samples_df = iter_samples_df.drop_duplicates()

            data_synth = pd.concat([data_synth, iter_samples_df], ignore_index=True)

            if len(data_synth) >= count:
                break

        data_synth = self.training_schema().adapt_dtypes(data_synth).head(count)

        return create_from_info(data_synth, self.data_info)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _safe_generate_time_series(
        self, gen_cbk: Callable, count: int, syn_schema: Schema, **kwargs: Any
    ) -> DataLoader:
        if self.data_info["data_type"] not in ["time_series", "time_series_survival"]:
            raise ValueError(
                f"Invalid data type for time series = {self.data_info['data_type']}"
            )
        constraints = syn_schema.as_constraints()

        data_synth = pd.DataFrame([], columns=self.training_schema().features())
        data_info = self.data_info
        offset = 0
        seq_offset = 0
        for it in range(self.sampling_patience):
            # sample
            if self.data_info["data_type"] == "time_series":
                static, temporal, observation_times, outcome = gen_cbk(
                    count - offset, **kwargs
                )
                loader = TimeSeriesDataLoader(
                    temporal_data=temporal,
                    observation_times=observation_times,
                    static_data=static,
                    outcome=outcome,
                    seq_offset=seq_offset,
                )
            elif self.data_info["data_type"] == "time_series_survival":
                static, temporal, observation_times, T, E = gen_cbk(
                    count - offset, **kwargs
                )
                loader = TimeSeriesSurvivalDataLoader(
                    temporal_data=temporal,
                    observation_times=observation_times,
                    static_data=static,
                    T=T,
                    E=E,
                    seq_offset=seq_offset,
                )

            # validate schema
            iter_samples_df = loader.dataframe()
            id_col = loader.info()["seq_id_feature"]

            iter_samples_df = self.training_schema().adapt_dtypes(iter_samples_df)

            if self.strict:
                iter_samples_df = constraints.match(iter_samples_df)

            if len(iter_samples_df) == 0:
                continue

            data_synth = pd.concat([data_synth, iter_samples_df], ignore_index=True)
            offset = len(data_synth[id_col].unique())
            seq_offset = max(data_synth[id_col].unique()) + 1

            if offset >= count:
                break

        data_synth = self.training_schema().adapt_dtypes(data_synth)
        return create_from_info(data_synth, data_info)

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

    def training_schema(self) -> Schema:
        """The internal schema"""
        if self._training_schema is None:
            raise RuntimeError("Fit the model first")

        return self._training_schema

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def plot(
        self,
        plt: Any,
        X: DataLoader,
        count: Optional[int] = None,
        plots: list = ["marginal", "associations", "tsne"],
        **kwargs: Any,
    ) -> Any:
        """Plot the real-synthetic distributions.

        Args:
            plt: output
            X: DataLoader.
                The reference dataset.

        Returns:
            self
        """
        X_syn = self.generate(count=count, **kwargs)

        if "marginal" in plots:
            plot_marginal_comparison(plt, X, X_syn)
        if "associations" in plots:
            plot_associations_comparison(plt, X, X_syn)
        if "tsne" in plots:
            plot_tsne(plt, X, X_syn)


class PluginLoader:
    """Plugin loading utility class.
    Used to load the plugins from the current folder.
    """

    @validate_arguments
    def __init__(self, plugins: list, expected_type: Type, categories: list) -> None:
        self._plugins: Dict[str, Type] = {}
        self._available_plugins = {}
        for plugin in plugins:
            stem = Path(plugin).stem.split("plugin_")[-1]
            self._available_plugins[stem] = plugin

        self._expected_type = expected_type
        self._categories = categories

    @validate_arguments
    def _load_single_plugin(self, plugin_name: str) -> None:
        """Helper for loading a single plugin"""
        plugin = Path(plugin_name)
        name = plugin.stem
        ptype = plugin.parent.name

        module_name = f"synthcity.plugins.{ptype}.{name}"

        failed = False
        for retry in range(2):
            try:
                if module_name in sys.modules:
                    mod = sys.modules[module_name]
                else:
                    spec = importlib.util.spec_from_file_location(module_name, plugin)
                    if not isinstance(spec.loader, Loader):
                        raise RuntimeError("invalid plugin type")

                    mod = importlib.util.module_from_spec(spec)
                    if module_name not in sys.modules:
                        sys.modules[module_name] = mod

                    spec.loader.exec_module(mod)
                cls = mod.plugin  # type: ignore
                failed = False
                break
            except BaseException as e:
                log.critical(f"load failed: {e}")
                failed = True

        if failed:
            log.critical(f"module {name} load failed")
            return

        log.debug(f"Loaded plugin {cls.type()} - {cls.name()}")
        self.add(cls.name(), cls)

    def list(self) -> List[str]:
        """Get all the available plugins."""
        all_plugins = list(self._plugins.keys()) + list(self._available_plugins.keys())
        plugins = []
        for plugin in all_plugins:
            if self.get_type(plugin).type() in self._categories:
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
    def load(self, buff: bytes) -> Any:
        """Load serialized plugin"""
        return Plugin.load(buff)

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
