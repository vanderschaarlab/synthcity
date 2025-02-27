import pytest
import pandas as pd
import numpy as np

from synthcity.plugins.core.models.syn_seq.syn_seq_encoder import Syn_SeqEncoder


def test_syn_seq_encoder_prepare():
    """
    Check .prepare() => sets up syn_order, method, variable_selection
    """
    df = pd.DataFrame({
        "col1": [10, 20, 30],
        "col2": [100, 200, 300],
        "col3": ["A","B","C"]
    })
    enc = Syn_SeqEncoder(
        syn_order=["col1","col2","col3"],
        method={"col2":"rf"},
        variable_selection={"col3":["col1"]},
        default_method="cart"
    )
    enc.prepare(df)

    info = enc.get_info()

    assert info["syn_order"] == ["col1","col2","col3"]
    assert info["method"]["col1"] == "swr"
    assert info["method"]["col2"] == "rf"
    assert info["method"]["col3"] == "cart"


def test_syn_seq_encoder_methods_on_small_df():
    """
    Minimal check that .prepare() doesn't crash on small DF
    """
    df = pd.DataFrame({"X":[1,2,3]})
    enc = Syn_SeqEncoder(
        syn_order=["X"],
        method={"X":"norm"},
        default_method="cart"
    )
    enc.prepare(df)
    info = enc.get_info()
    assert info["syn_order"] == ["X"]
    assert info["method"]["X"] == "norm"
    assert info["variable_selection"]["X"] == []
