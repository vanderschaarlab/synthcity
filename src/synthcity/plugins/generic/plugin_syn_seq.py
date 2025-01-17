
# stdlib
from pathlib import Path
from typing import Any, List, Optional, Union

# third party
import numpy as np
import pandas as pd

# Necessary packages
from pydantic import validate_arguments

# synthcity absolute
from synthcity.metrics.weighted_metrics import WeightedMetrics
from synthcity.plugins.core.dataloader import Syn_SeqDataLoader
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.models.syn_seq.syn_seq import Syn_Seq
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class Syn_SeqPlugin(Plugin):

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,

        **kwargs: Any
    ) -> None:
        super().__init__(

        )
 

    @staticmethod
    def name() -> str:
        return "syn_seq"

    @staticmethod
    def type() -> str:
        return "syn_seq"

    def _fit(self, X: Syn_SeqDataLoader, *args: Any, **kwargs: Any) -> "Syn_SeqPlugin":


        self.model = Syn_Seq(
            X.dataframe(),
 
        )
        self.model.fit(X.dataframe(), )

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> Syn_SeqDataLoader:

        return self._safe_generate(self.model.generate, count, syn_schema, )


plugin = Syn_SeqPlugin
