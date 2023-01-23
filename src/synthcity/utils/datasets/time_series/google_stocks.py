# stdlib
import io
from pathlib import Path
from typing import List, Tuple

# third party
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler

URL = "https://raw.githubusercontent.com/PacktPublishing/Learning-Pandas-Second-Edition/master/data/goog.csv"
df_path = Path(__file__).parent / "data/goog.csv"


class GoogleStocksDataloader:
    def __init__(self, seq_len: int = 10, as_numpy: bool = False) -> None:
        self.seq_len = seq_len
        self.as_numpy = as_numpy

    def load(
        self,
    ) -> Tuple[pd.DataFrame, List[pd.DataFrame], List, pd.DataFrame]:
        # Load Google Data
        if not df_path.exists():
            s = requests.get(URL).content
            df = pd.read_csv(io.StringIO(s.decode("utf-8")))

            df.to_csv(df_path, index=None)
        else:
            df = pd.read_csv(df_path)

        # Flip the data to make chronological data
        df = pd.DataFrame(df.values[::-1], columns=df.columns)
        T = (
            pd.to_datetime(df["Date"], infer_datetime_format=True)
            .astype(np.int64)
            .astype(np.float64)
            / 10**9
        )
        T = pd.Series(MinMaxScaler().fit_transform(T.values.reshape(-1, 1)).squeeze())

        df = df.drop(columns=["Date"])

        df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
        # Build dataset
        dataX = []
        dataT = []
        outcome = []

        # Cut data by sequence length
        for i in range(0, len(df) - self.seq_len - 1):
            df_seq = df.loc[i : i + self.seq_len - 1]
            horizons = T.loc[i : i + self.seq_len - 1]
            out = df["Open"].loc[i + self.seq_len]

            dataX.append(df_seq)
            dataT.append(horizons.values.tolist())
            outcome.append(out)

        # Mix Data (to make it similar to i.i.d)
        idx = np.random.permutation(len(dataX))

        temporal_data = []
        observation_times = []
        for i in range(len(dataX)):
            temporal_data.append(dataX[idx[i]])
            observation_times.append(dataT[idx[i]])

        if self.as_numpy:
            return (
                np.zeros((len(temporal_data), 0)),
                np.asarray(temporal_data, dtype=np.float32),
                np.asarray(observation_times),
                np.asarray(outcome, dtype=np.float32),
            )

        return (
            pd.DataFrame(np.zeros((len(temporal_data), 0))),
            temporal_data,
            observation_times,
            pd.DataFrame(outcome, columns=["Open_next"]),
        )
