# third party
import pandas as pd

# synthcity absolute
from synthcity.plugins.core.models.syn_seq.syn_seq_preprocess import SynSeqPreprocessor


def test_syn_seq_preprocess_basic() -> None:
    df = pd.DataFrame(
        {
            "date": ["2022-01-01", "2022-01-02", "N/A", "2022-01-05"],
            "sex": ["M", "F", "M", "F"],
            "income": [100, -8, 300, 400],
            "nociga": [0, -8, -8, 1],
        }
    )
    prep = SynSeqPreprocessor(
        user_dtypes={"date": "date", "sex": "category", "income": "numeric"},
        user_special_values={"income": [-8], "nociga": [-8]},
        max_categories=5,
    )
    df_pre = prep.preprocess(df)
    assert "income_cat" in df_pre.columns
    assert "nociga_cat" not in df_pre.columns
    df_post = prep.postprocess(df_pre)
    assert "income_cat" not in df_post.columns
    assert (df_post["income"] == -8).sum() == 1
    assert (df_post["nociga"] == -8).sum() == 2


def test_syn_seq_preprocess_auto_dtype() -> None:
    df = pd.DataFrame(
        {
            "A": ["cat1", "cat2", "cat1", "cat2"],
            "B": [100, 100, 100, 200],
            "C": [1, 2, 3, 4],
        }
    )
    prep = SynSeqPreprocessor(user_dtypes={}, user_special_values={}, max_categories=3)
    df_pre = prep.preprocess(df)
    assert prep.user_dtypes["A"] == "category"
    assert prep.user_dtypes["B"] == "category"
    assert prep.user_dtypes["C"] == "numeric"
    assert df_pre is not None
