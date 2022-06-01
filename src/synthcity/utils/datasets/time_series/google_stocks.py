# stdlib
from pathlib import Path
from typing import List, Optional, Tuple

# third party
import numpy as np
import pandas as pd

df_path = Path(__file__).parent / "data/goog.csv"


class GoogleStocksDataloader:
    def __init__(self, seq_len: int = 50) -> None:
        self.seq_len = seq_len

    def load(
        self,
    ) -> Tuple[Optional[pd.DataFrame], List[pd.DataFrame], Optional[pd.DataFrame]]:
        # Load Google Data
        df = pd.read_csv(df_path)

        # Flip the data to make chronological data
        df = pd.DataFrame(df.values[::-1], columns=df.columns)
        df = df.drop(columns=["date"])

        # Build dataset
        dataX = []
        outcome = []

        # Cut data by sequence length
        for i in range(0, len(df) - self.seq_len - 1):
            df_seq = df.loc[i : i + self.seq_len - 1]
            out = df["open"].loc[i + self.seq_len]

            dataX.append(df_seq)
            outcome.append(out)

        # Mix Data (to make it similar to i.i.d)
        idx = np.random.permutation(len(dataX))

        temporal_data = []
        for i in range(len(dataX)):
            temporal_data.append(dataX[idx[i]])

        return None, temporal_data, pd.DataFrame(outcome, columns=["open_next"])
