# stdlib
import urllib.error

# third party
import pandas as pd
from sklearn.datasets import load_diabetes
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

# synthcity absolute
from synthcity.utils.compression import compress_dataset, decompress_dataset


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
        header=None,
        names=[
            "frequency",
            "angle_of_attack",
            "chord_length",
            "free_stream_velocity",
            "suction_side_displacement_thickness",
            "scaled_sound_pressure_level",
        ],
    )
    df.columns = df.columns.astype(str)

    return df


def test_compression_sanity() -> None:
    df = load_diabetes(as_frame=True, return_X_y=True)[0]
    df["sex"] = df["sex"].astype(str)  # make categorical
    df["sex_dup"] = df["sex"]  # add redundant

    compressed_df, context = compress_dataset(df)

    assert len(compressed_df) == len(df)
    assert compressed_df.shape[1] > 0
    assert compressed_df.shape[1] <= df.shape[1]

    assert "encoders" in context
    assert "compressers" in context
    assert "compressers_categoricals" in context

    assert sorted(context["encoders"].keys()) == ["sex", "sex_dup"]
    assert len(context["compressers"]) == 1

    for col in context["compressers"]:
        assert "cols" in context["compressers"][col]
        assert "model" in context["compressers"][col]
        assert "min" in context["compressers"][col]
        assert "max" in context["compressers"][col]


def test_decompression_sanity() -> None:
    df = load_diabetes(as_frame=True, return_X_y=True)[0]
    df["sex"] = df["sex"].astype(str)  # make categorical
    df["sex_dup"] = df["sex"]  # add redundant

    compressed_df, context = compress_dataset(df)

    decompressed_df = decompress_dataset(compressed_df, context)

    assert sorted(df.columns.values) != sorted(compressed_df.columns.values)
    assert sorted(df.columns.values) == sorted(decompressed_df.columns.values)

    assert decompressed_df["sex"].dtype == "object"


def test_compression_sanity2() -> None:
    df = load_diabetes(as_frame=True, return_X_y=True)[0]
    compressed_df_orig, _ = compress_dataset(df)

    for col in df:
        df[f"{col}_duplicated"] = df[col]

    compressed_df, _ = compress_dataset(df)

    assert len(compressed_df) == len(df)
    assert compressed_df.shape[1] > 0
    assert compressed_df.shape[1] <= df.shape[1]

    assert compressed_df.shape == compressed_df_orig.shape


def test_compression_sanity_airfoil() -> None:
    df = get_airfoil_dataset()
    df["chord_length"] = df["chord_length"].astype(str)
    compressed_df, context = compress_dataset(df)

    assert len(compressed_df) == len(df)
    assert compressed_df.shape[1] > 0
    assert compressed_df.shape[1] <= df.shape[1]

    assert "encoders" in context
    assert "compressers" in context
    assert "compressers_categoricals" in context

    assert sorted(context["encoders"].keys()) == ["chord_length"]
    for key in context["encoders"]:
        assert context["encoders"][key].__class__.__name__ == "LabelEncoder"

    assert sorted(context["compressers"].keys()) == ["angle_of_attack"]
    for key in context["compressers"]:
        assert "cols" in context["compressers"][key]
        assert len(context["compressers"][key]["cols"]) > 0
        assert "model" in context["compressers"][key]

    assert sorted(context["compressers_categoricals"].keys()) == [
        "chord_length free_stream_velocity"
    ]
    for key in context["compressers_categoricals"]:
        assert "cols" in context["compressers_categoricals"][key]
        assert len(context["compressers_categoricals"][key]["cols"]) > 0
        assert "model" in context["compressers_categoricals"][key]


def test_decompression_sanity_airfoil() -> None:
    df = get_airfoil_dataset()
    df.columns = df.columns.astype(str)
    df["chord_length"] = df["chord_length"].astype(str)

    compressed_df, context = compress_dataset(df)

    decompressed_df = decompress_dataset(compressed_df, context)

    assert sorted(df.columns.values) != sorted(compressed_df.columns.values)
    assert sorted(df.columns.values) == sorted(decompressed_df.columns.values)

    assert decompressed_df["chord_length"].dtype == "object"
