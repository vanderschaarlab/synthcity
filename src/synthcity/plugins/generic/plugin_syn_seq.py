# File: plugin_syn_seq.py
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import validate_arguments

from synthcity.plugins.core.dataloader import Syn_SeqDataLoader
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.core.models.syn_seq.syn_seq import Syn_Seq


class Syn_SeqPlugin(Plugin):
    """
    A plugin that wraps the Syn_Seq aggregator. 
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        sampling_patience: int = 100,
        strict: bool = True,
        random_state: int = 0,
        device: Any = "cpu",  # not used but part of Plugin interface
        workspace: Path = Path("workspace"),
        compress_dataset: bool = False,
        sampling_strategy: str = "marginal",
        **kwargs: Any
    ) -> None:
        super().__init__(
            sampling_patience=sampling_patience,
            strict=strict,
            device=device,
            random_state=random_state,
            workspace=workspace,
            compress_dataset=compress_dataset,
            sampling_strategy=sampling_strategy,
        )
        # aggregator
        self.model: Optional[Syn_Seq] = None

    @staticmethod
    def name() -> str:
        return "syn_seq"

    @staticmethod
    def type() -> str:
        return "syn_seq"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        # return an empty or minimal set of hyperparameters for your plugin
        return []

    def _fit(self, X: Syn_SeqDataLoader, *args: Any, **kwargs: Any) -> "Syn_SeqPlugin":
        # initialize aggregator
        self.model = Syn_Seq(
            random_state=self.random_state,
            strict=self.strict,
            sampling_patience=self.sampling_patience,
        )
        # Fit aggregator
        self.model.fit(X)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> Syn_SeqDataLoader:
        if self.model is None:
            raise RuntimeError("Model not fitted")

        # maybe user passes in "rules" in kwargs
        rules = kwargs.get("rules", None)

        # call aggregator generate => pd.DataFrame
        df_syn = self.model.generate(nrows=count, rules=rules)

        # decode if needed
        df_syn_dec = self.model.decode(df_syn)

        # wrap in Syn_SeqDataLoader to respect the plugin framework
        # We'll do a minimal approach
        new_loader = Syn_SeqDataLoader(
            data=df_syn_dec,
            user_custom=None,  # no special config
            random_state=self.random_state,
            train_size=1.0,  # or something
            verbose=False
        )
        return new_loader

plugin = Syn_SeqPlugin