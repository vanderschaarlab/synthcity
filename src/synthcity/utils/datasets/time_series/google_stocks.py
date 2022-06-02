# stdlib
from pathlib import Path
from typing import List, Tuple

# third party
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df_path = Path(__file__).parent / "data/goog.csv"


class GoogleStocksDataloader:
    def __init__(self, seq_len: int = 10, as_numpy: bool = False) -> None:
        self.seq_len = seq_len
        self.as_numpy = as_numpy

    def load(
        self,
    ) -> Tuple[pd.DataFrame, List[pd.DataFrame], pd.DataFrame]:
        # Load Google Data
        df = pd.read_csv(df_path)

        # Flip the data to make chronological data
        df = pd.DataFrame(df.values[::-1], columns=df.columns)
        df = df.drop(columns=["date"])

        df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
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

        if self.as_numpy:
            return (
                np.zeros((len(temporal_data), 0)),
                np.asarray(temporal_data, dtype=np.float32),
                np.asarray(outcome, dtype=np.float32),
            )

        return (
            pd.DataFrame(np.zeros((len(temporal_data), 0))),
            temporal_data,
            pd.DataFrame(outcome, columns=["open_next"]),
        )
