# File: plugin_syn_seq.py

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.dataloader import DataLoader, create_from_info
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.core.models.syn_seq.syn_seq import Syn_Seq
from synthcity.utils.reproducibility import enable_reproducible_results


class Syn_SeqPlugin(Plugin):
    """
    The main plugin for the 'syn_seq' project. It overrides the parent's logic for:
      - fit(...) : to define our own sequence-based approach
      - generate(...) : to do column-by-column generation with optional 'rules'.

    The flow is:
      1. We build an original schema from the raw loader (unencoded).
      2. We explicitly call X.encode() => giving us `X_encoded` for training.
      3. We build a "training schema" from that encoded data.
      4. We pass X_encoded to our aggregator `Syn_Seq` for column-by-column fitting.
      5. For generation, we skip parent's `_generate()`, call aggregator's `.generate()`,
         decode back to original format, and optionally apply constraints if `strict=True`.
    """

    @staticmethod
    def name() -> str:
        return "syn_seq"

    @staticmethod
    def type() -> str:
        return "syn_seq"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List:
        # Provide any hyperparameter distributions if you want to tune them in AutoML.
        return []

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        random_state: int = 0,
        sampling_patience: int = 100,
        strict: bool = True,
        workspace: Path = Path("workspace"),
        compress_dataset: bool = False,
        sampling_strategy: str = "marginal",
        **kwargs: Any
    ) -> None:
        """
        Args:
            random_state: for reproducibility
            sampling_patience: maximum tries for re-generating rule-violating rows
            strict: whether to forcibly drop or keep rows that fail constraints
            workspace: path for caching (not used in this plugin specifically)
            compress_dataset: if True, drop redundant features before training
            sampling_strategy: "marginal" or "uniform" for how the schema is built
        """
        super().__init__(
            random_state=random_state,
            sampling_patience=sampling_patience,
            strict=strict,
            workspace=workspace,
            compress_dataset=compress_dataset,
            sampling_strategy=sampling_strategy,
        )

        self.model: Optional[Syn_Seq] = None
        self._schema: Optional[Schema] = None
        self._training_schema: Optional[Schema] = None
        self._data_encoders: Optional[Dict] = None  # store from X.encode()
        self.fitted = False

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: Union[DataLoader, pd.DataFrame], *args: Any, **kwargs: Any) -> "Syn_SeqPlugin":
        """
        Overridden fit: 
          - We create _schema from the unencoded loader
          - We do X.encode() => X_encoded
          - We create _training_schema from the encoded data
          - We pass X_encoded to aggregator Syn_Seq(...).fit(...)
        """
        enable_reproducible_results(self.random_state)

        if isinstance(X, pd.DataFrame):
            raise ValueError("syn_seq plugin requires a DataLoader, not a raw DataFrame")

        # 1) Original schema from the unencoded loader
        self._schema = Schema(
            data=X,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
        )

        # 2) Encode => X_encoded
        X_encoded, self._data_encoders = X.encode()

        # 3) Build training schema from X_encoded
        self._training_schema = Schema(
            data=X_encoded,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
        )

        # 4) aggregator
        self.model = Syn_Seq(
            random_state=self.random_state,
            strict=self.strict,
            sampling_patience=self.sampling_patience,
        )
        self.model.fit(X_encoded, *args, **kwargs)

        self.fitted = True
        return self

    def _generate(self, count: int, *args: Any, **kwargs: Any) -> DataLoader:
        """
        We do not use the parent's `_generate()` approach, so we override it with an error or pass.
        """
        raise NotImplementedError(
            "Syn_SeqPlugin doesn't use `_generate()`; see .generate() override instead."
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate(
        self,
        count: Optional[int] = None,
        constraints: Optional[Constraints] = None,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> DataLoader:
        """
        Overridden generate: 
          - Bypass parent's domain constraints => rely on aggregator's column-by-column generation
            plus optional 'rules' re-generation
          - Then decode back to original format
          - If strict => match() final constraints
        """
        if not self.fitted:
            raise RuntimeError("Must call .fit(...) first on syn_seq plugin")

        if count is None:
            # default: same length as training
            if self._schema is not None:
                count = self._schema.domain[self._schema.features()[0]].data.shape[0]
            else:
                raise ValueError("Cannot infer default count. Please specify nrows.")

        if random_state is not None:
            np.random.seed(random_state)

        # user-supplied rules dict
        rules = kwargs.pop("rules", None)

        # aggregator generate => "encoded" DataFrame
        raw_df = self.model.generate(nrows=count, rules=rules)

        # create a new DataLoader with the same "training_schema" info
        X_syn = create_from_info(raw_df, self._training_schema.info())

        # decode => revert cat-splitting, special values, etc.
        if X_syn.is_tabular() and self._data_encoders is not None:
            X_syn = X_syn.decode(self._data_encoders)

        # final constraints if strict
        if constraints is not None and self.strict:
            X_syn = X_syn.match(constraints)

        return X_syn

plugin = Syn_SeqPlugin