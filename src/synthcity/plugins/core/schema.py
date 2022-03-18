# third party
import pandas as pd
from pydantic import validate_arguments


class Schema:
    @validate_arguments
    def __init__(self, X: pd.DataFrame) -> None:
        pass
