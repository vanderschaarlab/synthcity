# stdlib
import io
from pathlib import Path
from typing import List, Tuple

# third party
import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

URL = "https://raw.githubusercontent.com/autonlab/auton-survival/cf583e598ec9ab92fa5d510a0ca72d46dfe0706f/dsm/datasets/pbc2.csv"
df_path = Path(__file__).parent / "data/pbc2.csv"


class PBCDataloader:
    """Helper function to load and preprocess the PBC dataset

    The Primary biliary cirrhosis (PBC) Dataset [1] is well known
    dataset for evaluating survival analysis models with time
    dependent covariates.

    Args:

    - seq_len: sequence length of the time-series

    Returns:
    - data: generated data

    """

    def __init__(
        self,
        seq_len: int = 10,
        as_numpy: bool = False,
    ) -> None:
        self.seq_len = seq_len
        self.as_numpy = as_numpy

    def _load_pbc_dataset(self, sequential: bool = True) -> Tuple:
        """Helper function to load and preprocess the PBC dataset
        The Primary biliary cirrhosis (PBC) Dataset [1] is well known
        dataset for evaluating survival analysis models with time
        dependent covariates.
        Parameters
        ----------
        sequential: bool
          If True returns a list of np.arrays for each individual.
          else, returns collapsed results for each time step. To train
          recurrent neural models you would typically use True.
        References
        ----------
        [1] Fleming, Thomas R., and David P. Harrington. Counting processes and
        survival analysis. Vol. 169. John Wiley & Sons, 2011.
        """
        if not df_path.exists():
            s = requests.get(URL).content
            data = pd.read_csv(io.StringIO(s.decode("utf-8")))
            data.to_csv(df_path)
        else:
            data = pd.read_csv(df_path)

        data["histologic"] = data["histologic"].astype(str)
        dat_cat = data[
            ["drug", "sex", "ascites", "hepatomegaly", "spiders", "edema", "histologic"]
        ]
        dat_num = data[
            [
                "serBilir",
                "serChol",
                "albumin",
                "alkaline",
                "SGOT",
                "platelets",
                "prothrombin",
            ]
        ]
        age = data["age"] + data["years"]

        x1 = pd.get_dummies(dat_cat).values
        x2 = dat_num.values
        x3 = age.values.reshape(-1, 1)
        x = np.hstack([x1, x2, x3])

        time = (data["years"] - data["year"]).values
        event = data["status2"].values

        x = SimpleImputer(missing_values=np.nan, strategy="mean").fit_transform(x)
        x_ = StandardScaler().fit_transform(x)

        if not sequential:
            return x_, time, event
        else:
            x, t, e = [], [], []
            for id_ in sorted(list(set(data["id"]))):
                x.append(x_[data["id"] == id_])
                t.append(time[data["id"] == id_])
                e.append(event[data["id"] == id_])
            return x, t, e

    def load(
        self,
    ) -> Tuple[pd.DataFrame, List[pd.DataFrame], pd.DataFrame]:
        # Initialize the output

        temporal_data, t, e = self._load_pbc_dataset()

        static_data = pd.DataFrame(np.zeros((len(temporal_data), 0)))
        outcome = pd.concat(
            [pd.Series(t, name="time_to_event"), pd.Series(e, name="event")], axis=1
        )

        if self.as_numpy:
            return (
                np.asarray(static_data),
                np.asarray(temporal_data),
                np.asarray(outcome),
            )

        return static_data, temporal_data, outcome
