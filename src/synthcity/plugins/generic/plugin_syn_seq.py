# stdlib
from typing import Any, List, Optional, cast

# third party
import pandas as pd
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader, Syn_SeqDataLoader
from synthcity.plugins.core.models.syn_seq.syn_seq import Syn_Seq
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class Syn_SeqPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.generic.plugin_syn_seq.Syn_SeqPlugin
        :parts: 1

    Synthetic Sequence Generation Plugin.

    Args:
        workspace: Path.
            Optional Path for caching intermediary results.
        compress_dataset: bool. Default = False.
            Drop redundant features before training the generator.
        sampling_patience: int.
            Max inference iterations to wait for the generated data to match the training schema.
        random_state: int.
            random_state used
        sampling_strategy: str.
            Sampling strategy to use for generating synthetic data. Options are 'marginal' or 'joint'.
            Default is 'marginal'.


    Example:
        >>> from sklearn.datasets import load_iris
        >>> from synthcity.plugins import Plugins
        >>>
        >>> X, y = load_iris(as_frame = True, return_X_y = True)
        >>> X["target"] = y
        >>>
        >>> plugin = Plugins().get("syn_seq")
        >>> plugin.fit(X)
        >>>
        >>> plugin.generate(50)

    """

    @staticmethod
    def name() -> str:
        return "syn_seq"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List:
        return []

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        sampling_patience: int = 100,
        random_state: int = 0,
        compress_dataset: bool = False,
        sampling_strategy: str = "marginal",
        **kwargs: Any
    ) -> None:
        super().__init__(
            random_state=random_state,
            sampling_patience=sampling_patience,
            compress_dataset=compress_dataset,
            sampling_strategy=sampling_strategy,
        )
        self.model: Optional[Syn_Seq] = None

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "Syn_SeqPlugin":
        if isinstance(X, pd.DataFrame):
            X = Syn_SeqDataLoader(X)
        self.model = Syn_Seq(
            random_state=self.random_state,
            sampling_patience=self.sampling_patience,
        )

        # cast explicitly to Syn_Seq to make sure mypy doesn't think it can be None
        cast(Syn_Seq, self.model).fit_col(
            X, self._data_encoders, self.data_info, *args, **kwargs
        )
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> DataLoader:
        if self.model is None:
            raise RuntimeError("The model must be fitted before generating data.")
        df_syn = self.model.generate_col(count, self._data_encoders, **kwargs)
        df_syn = syn_schema.adapt_dtypes(df_syn)
        return Syn_SeqDataLoader(
            df_syn, user_custom=self.data_info.get("user_custom", {}), verbose=False
        )


plugin = Syn_SeqPlugin
