# third party
from gansfer.utils.compression import compress_dataset
from sklearn.datasets import load_diabetes


def test_compression_sanity() -> None:
    df = load_diabetes(as_frame=True, return_X_y=True)[0]

    compressed_df = compress_dataset(df)

    assert len(compressed_df) == len(df)
    assert compressed_df.shape[1] > 0
    assert compressed_df.shape[1] <= df.shape[1]


def test_compression_sanity2() -> None:
    df = load_diabetes(as_frame=True, return_X_y=True)[0]
    compressed_df_orig = compress_dataset(df)

    for col in df:
        df[f"{col}_duplicated"] = df[col]

    compressed_df = compress_dataset(df)
    print(df.shape, compressed_df.shape)

    assert len(compressed_df) == len(df)
    assert compressed_df.shape[1] > 0
    assert compressed_df.shape[1] <= df.shape[1]

    assert compressed_df.shape == compressed_df_orig.shape
