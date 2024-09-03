# stdlib
import urllib.error

# third party
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed


@retry(
    stop=stop_after_attempt(5),  # Retry up to 5 times
    wait=wait_fixed(2),  # Wait 2 seconds between retries
    retry=retry_if_exception_type(urllib.error.HTTPError),  # Retry on HTTPError
)
def get_airfoil_dataset() -> pd.DataFrame:
    """
    Downloads the Airfoil Self-Noise dataset and returns it as a DataFrame.

    Returns:
        pd.DataFrame: The Airfoil Self-Noise dataset.
    """
    # Read the dataset from the URL
    df = pd.read_csv(
        "https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip",
        sep="\t",
        engine="python",
    )
    df.columns = df.columns.astype(str)

    return df
