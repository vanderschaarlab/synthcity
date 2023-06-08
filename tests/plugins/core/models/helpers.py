# third party
import pandas as pd


def get_airfoil_dataset() -> pd.DataFrame:
    df = pd.read_csv(
        "https://www.neuraldesigner.com/files/datasets/airfoil_self_noise.csv",
        # "https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip", # TODO: change to this source?
        sep=";",
        engine="python",
    )
    df.columns = df.columns.astype(str)

    return df
