# synthcity/plugins/syn_seq/plugin_syn_seq.py

from typing import Any, Optional

import pandas as pd

# synthcity absolute
from synthcity.plugins.core.base_plugin import Plugin
from .syn_seq_dataloader import Syn_SeqDataLoader


class Syn_SeqPlugin(Plugin):
    """
    A plugin example for the 'syn_seq' approach,
    demonstrating how to fit and generate synthetic data with Syn_Seq logic.
    """

    def __init__(
        self,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(plugin_name="syn_seq")
        self.random_state = random_state
        self.params = kwargs
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
        # e.g. encode => (encoded_loader, {"syn_seq_encoder": encoder})
        encoded_loader, encoders = dataloader.encode()

        df = encoded_loader.dataframe()
        if df.empty:
            raise ValueError("No data to train on in Syn_SeqPlugin.")

        # TODO: example model training
        # self.model = someModel(...)
        # self.model.fit(df, ...)

        self._model_trained = True
        self._encoders = encoders
        return self

    def generate(self, count: int = 10, *args, **kwargs) -> Syn_SeqDataLoader:
        if not self._model_trained:
            raise RuntimeError("Syn_SeqPlugin: fit must be called before generate().")

        # dummy synthetic df for example
        synthetic_df = pd.DataFrame({
            "feature1": [1] * count,
            "feature2": [0] * count,
        })

        # if we want decode
        if "syn_seq_encoder" in self._encoders:
            # encoder = self._encoders["syn_seq_encoder"]
            # synthetic_df = encoder.inverse_transform(synthetic_df)
            pass

        # create Syn_SeqDataLoader
        syn_loader = Syn_SeqDataLoader(
            data=synthetic_df,
            target_order=list(synthetic_df.columns),
        )
        return syn_loader
