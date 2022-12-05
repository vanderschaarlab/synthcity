# third party
import pandas as pd


def get_airfoil_dataset() -> pd.DataFrame:
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
        header=None,
        sep="\\t",
        engine="python",
    )
    df.columns = df.columns.astype(str)

    return df
