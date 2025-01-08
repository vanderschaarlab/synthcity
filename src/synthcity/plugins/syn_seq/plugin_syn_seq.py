# synthcity/plugins/syn_seq/plugin_syn_seq.py

"""
Plugin: Syn_SeqPlugin
---------------------

A plugin for a sequential regression-based approach, inspired by R's `synthpop`
but extended to fit the synthcity architecture. It uses a SynSeqSynthesizer
internally to manage column-by-column (or feature-by-feature) synthetic data
generation with flexible method assignments (CART, ctree, logreg, etc.).

Core steps:
    1) Take a Syn_SeqDataLoader input (with special value mappings, thresholding, etc.).
    2) Encode the data using Syn_SeqEncoder (if no encoder is already provided).
    3) Pass the encoded data to the SynSeqSynthesizer for fitting.
    4) During generate(), produce samples by calling the synthesizer, 
       then optionally decode them via the stored encoder.
    5) Return the samples as a new Syn_SeqDataLoader.

Usage Example:
    >>> from synthcity.plugins.syn_seq.syn_seq_dataloader import Syn_SeqDataLoader
    >>> from synthcity.plugins.syn_seq.plugin_syn_seq import Syn_SeqPlugin
    >>>
    >>> # Suppose df is a pandas DataFrame with columns ["A","B","C",...]
    >>> # We define a syn_order, special values, etc.
    >>> data_loader = Syn_SeqDataLoader(
    ...     data=df,
    ...     syn_order=["A","B","C",...],
    ...     columns_special_values={"A":[999]}, 
    ...     max_categories=20,
    ... )
    >>>
    >>> plugin = Syn_SeqPlugin(
    ...     random_state=42,
    ...     visit_sequence=["A","B","C"],
    ...     method_map={"A":"CART","B":"logreg","C":"polyreg"}
    ... )
    >>>
    >>> # Fit the plugin
    >>> plugin.fit(data_loader)
    >>>
    >>> # Generate synthetic data
    >>> syn_data = plugin.generate(count=100)
    >>> syn_df = syn_data.dataframe()
    >>> print(syn_df.head())
"""

from typing import Any, Optional

import pandas as pd

# synthcity absolute
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.utils.reproducibility import enable_reproducible_results
from synthcity.plugins.core.dataloader import DataLoader

# local
from .syn_seq_dataloader import Syn_SeqDataLoader
from .syn_seq_synthesizer import SynSeqSynthesizer


class Syn_SeqPlugin(Plugin):
    """
    A plugin for the 'syn_seq' approach (sequential regression style),
    similar in spirit to R's `synthpop` but adapted for synthcity. 
    This plugin uses a SynSeqSynthesizer as the internal logic.

    Arguments:
        random_state: int
            Random seed.
        visit_sequence: list
            The sequence of columns for iterative modeling. 
        method_map: dict
            Mapping of columns to method names, e.g. {"colA":"CART","colB":"logreg"}.
        use_rules: bool
            If True, the synthesizer might apply extra rules or constraints.
        **kwargs: optional
            Additional arguments for the base Plugin or the synthesizer.

    Key steps:
        - In `fit()`: we encode the incoming DataLoader (if needed), 
          pass the encoded DataFrame to SynSeqSynthesizer, 
          and store encoders for decoding the output.
        - In `generate()`: we call the synthesizer to produce new samples, 
          optionally decode them, and return as Syn_SeqDataLoader.
    """

    def __init__(
        self,
        random_state: Optional[int] = None,
        visit_sequence: Optional[list] = None,
        method_map: Optional[dict] = None,
        use_rules: bool = False,
        **kwargs: Any,
    ):
        # We call the base Plugin constructor with possible 
        # general plugin arguments (like 'strict', 'workspace', etc. in kwargs).
        super().__init__(
            random_state=(random_state if random_state is not None else 0),
            **kwargs
        )

        self.visit_sequence = visit_sequence or []
        self.method_map = method_map or {}
        self.use_rules = use_rules

        # Store any additional parameters
        self.params = kwargs

        # The core sequential regression synthesizer logic
        self.synthesizer = SynSeqSynthesizer(
            visit_sequence=self.visit_sequence,
            method_map=self.method_map,
            use_rules=self.use_rules,
            **kwargs  # if we want to forward any other config
        )

        self._model_trained = False
        self._encoders = {}

    @staticmethod
    def name() -> str:
        """Returns the name of the plugin."""
        return "syn_seq"

    @staticmethod
    def type() -> str:
        """Returns the type/category of the plugin."""
        # Could be "generic" or a new custom category. We'll keep it "generic" for now.
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> list:
        """
        For AutoML or hyperparameter search, define the tuneable parameters.
        If none, return an empty list.
        """
        return []

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "Syn_SeqPlugin":
        """
        The internal `_fit` method, called by Plugin.fit(...) after 
        data schema/encoding steps are handled.
        """
        # Convert to DataFrame
        df = X.dataframe()
        if df.empty:
            raise ValueError("No data to train on in Syn_SeqPlugin.")

        # Call the internal synthesizer fit
        self.synthesizer.fit(df, *args, **kwargs)
        self._model_trained = True
        return self

    def _generate(
        self,
        count: int,
        syn_schema: Schema,
        **kwargs: Any,
    ) -> DataLoader:
        """
        The internal `_generate` method, 
        typically using `_safe_generate` from the base Plugin to handle constraints.
        """
        if not self._model_trained:
            raise RuntimeError("Syn_SeqPlugin: must fit() first.")

        # We'll let `_safe_generate` handle the generation pipeline, 
        # calling self.synthesizer.generate(...) as a callback.
        return self._safe_generate(
            gen_cbk=self.synthesizer.generate,
            count=count,
            syn_schema=syn_schema,
            **kwargs
        )

    def fit(
        self,
        dataloader: Syn_SeqDataLoader, 
        *args: Any, 
        **kwargs: Any
    ) -> "Syn_SeqPlugin":
        """
        Public fit entrypoint:
          1) We encode the incoming Syn_SeqDataLoader (if no external encoder is given).
          2) We call super().fit(...) so that it will invoke `_fit`.
        """
        if not isinstance(dataloader, Syn_SeqDataLoader):
            raise TypeError("Syn_SeqPlugin expects a Syn_SeqDataLoader")

        # encode => returns (new_dataloader, encoders)
        encoded_loader, encoders = dataloader.encode()
        df = encoded_loader.dataframe()
        if df.empty:
            raise ValueError("No data to train on in Syn_SeqPlugin.")

        self._encoders = encoders  # store for decode in generate

        # Now proceed with the parent's fit logic 
        # (which calls `_fit` internally).
        super().fit(encoded_loader, *args, **kwargs)
        return self

    def generate(
        self,
        count: int = 10,
        *args: Any,
        **kwargs: Any
    ) -> Syn_SeqDataLoader:
        """
        Generate synthetic samples:
          1) We use the parent's generate logic (which calls `_generate`),
          2) decode if needed,
          3) wrap in Syn_SeqDataLoader for the user.
        """
        if not self._model_trained:
            raise RuntimeError("Syn_SeqPlugin: must fit() before generate()")

        # Let the base plugin handle constraints, `_generate` call, etc.
        gen_data: DataLoader = super().generate(count=count, *args, **kwargs)

        # Optionally decode 
        syn_df = gen_data.dataframe()
        if "syn_seq_encoder" in self._encoders:
            encoder = self._encoders["syn_seq_encoder"]
            syn_df = encoder.inverse_transform(syn_df)

        # Return as a Syn_SeqDataLoader 
        # (keeping the same columns order or the original order).
        return Syn_SeqDataLoader(
            data=syn_df,
            syn_order=list(syn_df.columns),  # or the original syn_order if needed
        )
