# =====================
# file: plugin_syn_seq.py
# =====================
from typing import Any, Optional
import pandas as pd

# synthcity absolute
from synthcity.plugins.core.base_plugin import Plugin
from .syn_seq_dataloader import Syn_SeqDataLoader
# 우리가 만든 Synthesizer import
from .syn_seq_synthesizer import SynSeqSynthesizer

class Syn_SeqPlugin(Plugin):
    """
    A plugin example for the 'syn_seq' approach,
    that internally uses SynSeqSynthesizer for the actual R-synthpop-like logic.
    """

    def __init__(
        self,
        random_state: Optional[int] = None,
        visit_sequence: Optional[list] = None,
        method_map: Optional[dict] = None,
        use_rules: bool = False,
        **kwargs: Any,
    ):
        super().__init__(plugin_name="syn_seq")
        
        self.random_state = random_state
        self.visit_sequence = visit_sequence or []
        self.method_map = method_map or {}
        self.use_rules = use_rules
        self.params = kwargs

        # 핵심: plugin 안에 우리가 만든 Synthesizer를 멤버로 둠
        self.synthesizer = SynSeqSynthesizer(
            visit_sequence=self.visit_sequence,
            method_map=self.method_map,
            use_rules=self.use_rules
            # 필요하면 **kwargs
        )

        self._model_trained = False
        self._encoders = {}

    @staticmethod
    def name() -> str:
        return "syn_seq"

    @staticmethod
    def type() -> str:
        # e.g. Plugins(categories=["generic","syn_seq"]) ...
        return "generic"

    def fit(self, dataloader: Syn_SeqDataLoader, *args, **kwargs) -> "Syn_SeqPlugin":
        """
        1) encode to get DataFrame
        2) pass to self.synthesizer.fit
        """
        encoded_loader, encoders = dataloader.encode()

        df = encoded_loader.dataframe()
        if df.empty:
            raise ValueError("No data to train on in Syn_SeqPlugin.")

        # call synthesizer's fit
        self.synthesizer.fit(df, *args, **kwargs)

        self._model_trained = True
        self._encoders = encoders
        return self

    def generate(self, count: int = 10, *args, **kwargs) -> Syn_SeqDataLoader:
        if not self._model_trained:
            raise RuntimeError("Syn_SeqPlugin: fit must be called before generate().")

        # call synthesizer's generate
        syn_df = self.synthesizer.generate(count=count, **kwargs)

        # encoder의 역변환을 원한다면(필요하다면) inverse_transform
        if "syn_seq_encoder" in self._encoders:
            encoder = self._encoders["syn_seq_encoder"]
            syn_df = encoder.inverse_transform(syn_df)

        # wrap DataFrame into Syn_SeqDataLoader
        syn_loader = Syn_SeqDataLoader(
            data=syn_df,
            target_order=list(syn_df.columns), # or original?
        )
        return syn_loader
