# stdlib
from pathlib import Path

# third party
import numpy as np
import pandas as pd

df_path = Path(__file__).parent / "data/goog.csv"


class GoogleStocksDataloader:
    def __init__(self, seq_len: int) -> None:
        self.seq_len = seq_len

    def load(self) -> np.ndarray:
        # Load Google Data
        df = pd.read_csv(df_path)

        # Flip the data to make chronological data
        x = df.values[::-1]

        # Build dataset
        dataX = []

        # Cut data by sequence length
        for i in range(0, len(x) - self.seq_len):
            _x = x[i : i + self.seq_len]
            dataX.append(_x)

        # Mix Data (to make it similar to i.i.d)
        idx = np.random.permutation(len(dataX))

        outputX = []
        for i in range(len(dataX)):
            outputX.append(dataX[idx[i]])

        return np.asarray(outputX)
