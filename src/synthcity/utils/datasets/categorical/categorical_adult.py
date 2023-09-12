# stdlib
import io
from pathlib import Path
from typing import Union

# third party
import numpy as np
import pandas as pd
import requests

URL = "https://raw.githubusercontent.com/ryan112358/private-pgm/master/data/adult.csv"
df_path = Path(__file__).parent / "data/adult.csv"


class CategoricalAdultDataloader:
    def __init__(self, as_numpy: bool = False) -> None:
        self.as_numpy = as_numpy

    def load(
        self,
    ) -> Union[pd.DataFrame, np.ndarray]:
        # Load Google Data
        if not df_path.exists():
            s = requests.get(URL, timeout=5).content
            df = pd.read_csv(io.StringIO(s.decode("utf-8")))

            df.to_csv(df_path, index=None)
        else:
            df = pd.read_csv(df_path)

        if self.as_numpy:
            return df.to_numpy()
        return df
