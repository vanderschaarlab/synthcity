# stdlib
from pathlib import Path
from typing import Tuple

# third party
import numpy as np
import pandas as pd
from lifelines.datasets import load_rossi
from pycox import datasets
from sklearn.preprocessing import LabelEncoder
from sksurv.datasets import load_aids, load_flchain, load_gbsg2, load_whas500


def get_dataset(name: str) -> Tuple[pd.DataFrame, str, str, list]:
    data_folder = Path("data")
    data_folder.mkdir(parents=True, exist_ok=True)

    if name == "metabric":
        raw_df = pd.read_csv("data/metabric.csv")
        X = raw_df.drop(columns=["overall_survival_months", "overall_survival"])
        T = raw_df["overall_survival_months"]
        E = raw_df["overall_survival"]

        df = X.copy()
        df["event"] = E
        df["duration"] = T
    elif name == "support":
        df = datasets.support.read_df()
    elif name == "gbsg":
        df = datasets.gbsg.read_df()
    elif name == "rossi":
        df = load_rossi()
        df = df.rename(columns={"week": "duration", "arrest": "event"})
    elif name == "aids":
        X, Y = load_aids()
        Y_unp = np.array(Y, dtype=[("event", "int"), ("duration", "float")])
        df = X.copy()
        df["event"] = Y_unp["event"]
        df["duration"] = Y_unp["duration"]
    elif name == "flchain":
        X, Y = load_flchain()
        Y_unp = np.array(Y, dtype=[("event", "int"), ("duration", "float")])
        df = X.copy()
        df["event"] = Y_unp["event"]
        df["duration"] = Y_unp["duration"]
    elif name == "gbsg2":
        X, Y = load_gbsg2()
        Y_unp = np.array(Y, dtype=[("event", "int"), ("duration", "float")])
        df = X.copy()
        df["event"] = Y_unp["event"]
        df["duration"] = Y_unp["duration"]
    elif name == "whas500":
        X, Y = load_whas500()
        Y_unp = np.array(Y, dtype=[("event", "int"), ("duration", "float")])
        df = X.copy()
        df["event"] = Y_unp["event"]
        df["duration"] = Y_unp["duration"]

    for col in df.columns:
        if df[col].dtype.name in ["object", "category"]:
            df[col] = LabelEncoder().fit_transform(df[col])

    duration_col = "duration"
    event_col = "event"

    df = df.fillna(0)

    T = df[duration_col]

    time_horizons = np.linspace(T.min(), T.max(), num=5)[1:-1].tolist()

    return df, duration_col, event_col, time_horizons
