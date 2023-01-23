# stdlib
import io
from pathlib import Path
from typing import List, Tuple

# third party
import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
            data.to_csv(df_path, index=None)
        else:
            data = pd.read_csv(df_path)

        data["time"] = data["years"] - data["year"]
        data = data.sort_values(by="time")
        data["histologic"] = data["histologic"].astype(str)
        dat_cat = data[
            ["drug", "sex", "ascites", "hepatomegaly", "spiders", "edema", "histologic"]
        ].copy()
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

        for col in dat_cat.columns:
            dat_cat[col] = LabelEncoder().fit_transform(dat_cat[col])

        x1 = dat_cat
        x2 = dat_num
        x3 = pd.Series(age, name="age")
        x = pd.concat([x1, x2, x3], axis=1)

        time = data["time"]
        event = data["status2"]

        x = pd.DataFrame(
            SimpleImputer(missing_values=np.nan, strategy="mean").fit_transform(x),
            columns=x.columns,
        )
        scaled_cols = []
        for col in x.columns:
            if len(x[col].unique()) < 15:
                continue
            scaled_cols.append(col)

        x_ = x.copy()
        x_[scaled_cols] = pd.DataFrame(
            StandardScaler().fit_transform(x[scaled_cols]),
            columns=scaled_cols,
            index=data.index,
        )

        x_static, x_temporal, t, e = [], [], [], []
        t_ext, e_ext = [], []

        temporal_cols = [
            "drug",
            "ascites",
            "hepatomegaly",
            "spiders",
            "edema",
            "histologic",
            "serBilir",
            "serChol",
            "albumin",
            "alkaline",
            "SGOT",
            "platelets",
            "prothrombin",
            "age",
        ]

        static_cols = [
            "sex",
        ]

        for id_ in sorted(list(set(data["id"]))):
            patient = x_[data["id"] == id_]

            patient_static = patient[static_cols]
            x_static.append(patient_static.values[0].tolist())

            patient_temporal = patient[temporal_cols]
            patient_temporal.index = time[data["id"] == id_].values

            if self.as_numpy:
                x_temporal.append(patient_temporal.values)
            else:
                x_temporal.append(patient_temporal)

            events = event[data["id"] == id_].values
            times = time[data["id"] == id_].values
            evt = np.amax(events)
            if evt == 0:
                pos = np.max(np.where(events == evt))  # last censored
            else:
                pos = np.min(np.where(events == evt))  # first event

            t.append(times[pos])
            e.append(evt)

            t_ext.append(times)
            e_ext.append(events)

        if self.as_numpy:
            x_static = np.asarray(x_static)
            x_temporal = np.array(x_temporal, dtype=object)
            t = np.asarray(t)
            e = np.asarray(e)
        else:
            x_static = pd.DataFrame(x_static, columns=static_cols)
            t = pd.Series(t, name="time_to_event")
            e = pd.Series(e, name="event")

        return (
            x_static,
            x_temporal,
            t,
            e,
            t_ext,
            e_ext,
        )

    def load(
        self,
    ) -> Tuple[pd.DataFrame, List[pd.DataFrame], List, pd.DataFrame]:
        # Initialize the output

        static_data, temporal_data, t, e, t_ext, e_ext = self._load_pbc_dataset()
        outcome = (t, e)

        return static_data, temporal_data, t_ext, outcome
