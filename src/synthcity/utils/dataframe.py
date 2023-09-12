# third party
import pandas as pd


def constant_columns(dataframe: pd.DataFrame) -> list:
    """
    Find constant value columns in a pandas dataframe.
    """
    return discrete_columns(dataframe, 1)


def discrete_columns(
    dataframe: pd.DataFrame, max_classes: int = 10, return_counts: bool = False
) -> list:
    """
    Find columns containing discrete values in a pandas dataframe.
    """
    return [
        (col, cnt) if return_counts else col
        for col, vals in dataframe.items()
        for cnt in [vals.nunique()]
        if cnt <= max_classes
    ]
