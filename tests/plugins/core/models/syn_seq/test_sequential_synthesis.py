import numpy as np
import pandas as pd
import pytest

# Import the sequential synthesizer and the METHOD_MAP from syn_seq.py
from synthcity.plugins.core.models.syn_seq.syn_seq import Syn_Seq, METHOD_MAP, NUMERIC_MARKER

# --- Define dummy functions for a “cart” (and “swr”) synthesis method ---
def dummy_syn_fit(y, X, random_state=0):
    # Dummy “fit” function: simply returns a dummy model marker.
    return "dummy_model"

def dummy_generate(dummy_model, Xsyn):
    # Dummy “generate” function: for each numeric input value, generate synthetic output = input + 100.
    # Here we assume Xsyn is a numeric numpy array.
    return Xsyn.flatten() + 100

# Patch the METHOD_MAP so that the "cart" (and "swr") method uses our dummy functions.
METHOD_MAP["cart"] = (dummy_syn_fit, dummy_generate)
METHOD_MAP["swr"] = (dummy_syn_fit, dummy_generate)

# --- Create a dummy DataLoader-like class that provides a DataFrame ---
class DummyLoader:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def dataframe(self):
        return self._df

# --- Optionally, a dummy label encoder for tests that need one ---
class DummyLabelEncoder:
    def __init__(self, classes):
        self.classes_ = np.array(classes)

# --- Fixtures ---
@pytest.fixture
def sample_df():
    # A simple DataFrame with two numeric columns.
    return pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50]
    })

@pytest.fixture
def dummy_loader(sample_df):
    return DummyLoader(sample_df)

@pytest.fixture
def loader_info():
    # Define synthesis order and variable selection.
    return {
        "syn_order": ["A", "B"],
        "method": {"B": "cart"},
        "variable_selection": {"B": ["A"]}
    }

# --- Tests for the sequential synthesizer ---

def test_sequential_synthesis_basic(dummy_loader, loader_info, sample_df):
    """
    Test a basic sequential synthesis:
      - Use a DataFrame with two numeric columns, A and B.
      - Define synthesis order (["A", "B"]), with column B synthesized using method "cart"
        and its predictors set as ["A"].
      - Manually simulate storing first‑column values and set a dummy fitted model for B.
      - Generate synthetic data and check that column B is generated as “A + 100.”
    """
    syn_seq = Syn_Seq(random_state=42, sampling_patience=100)
    syn_seq.fit_col(dummy_loader, label_encoder={}, loader_info=loader_info)
    
    # Verify internal state was set as expected.
    assert syn_seq._syn_order == ["A", "B"]
    assert syn_seq._method_map == {"B": "cart"}
    assert syn_seq._varsel == {"B": ["A"]}
    
    # Simulate storing first‑column values and a fitted model for column "B".
    syn_seq._first_col_values = {"A": np.array(sample_df["A"])}
    syn_seq._col_models["B"] = {"name": "cart", "fitted_model": "dummy_model"}
    syn_seq._model_trained = True
    
    # Generate synthetic data (count = 3)
    gen_df = syn_seq.generate_col(3, label_encoder={})
    
    # Check that the generated DataFrame has the expected columns and rows.
    assert list(gen_df.columns) == ["A", "B"]
    assert len(gen_df) == 3
    # For each generated row, B should equal A + 100.
    for i in range(3):
        a_val = gen_df.loc[i, "A"]
        b_val = gen_df.loc[i, "B"]
        np.testing.assert_almost_equal(b_val, a_val + 100)

def test_sequential_synthesis_empty_data():
    """
    Test that calling fit_col on a DataLoader whose DataFrame is empty
    raises a ValueError.
    """
    empty_df = pd.DataFrame(columns=["A", "B"])
    loader = DummyLoader(empty_df)
    loader_info = {
        "syn_order": ["A", "B"],
        "method": {"B": "cart"},
        "variable_selection": {"B": ["A"]}
    }
    syn_seq = Syn_Seq()
    with pytest.raises(ValueError, match="No data => cannot fit Syn_Seq aggregator"):
        syn_seq.fit_col(loader, label_encoder={}, loader_info=loader_info)

@pytest.mark.parametrize("count", [0, -5])
def test_sequential_synthesis_zero_or_negative_count(count):
    """
    Test that calling generate_col with count <= 0 returns an empty DataFrame
    with the expected column names.
    """
    df = pd.DataFrame({"A": [1, 2, 3], "B": [10, 20, 30]})
    syn_seq = Syn_Seq()
    syn_seq._syn_order = ["A", "B"]
    syn_seq._model_trained = True
    gen_df = syn_seq.generate_col(count, label_encoder={})
    assert gen_df.empty
    assert list(gen_df.columns) == ["A", "B"]

def test_sequential_synthesis_not_fitted():
    """
    Test that calling generate_col before the model is marked as trained
    raises a RuntimeError.
    """
    syn_seq = Syn_Seq()
    with pytest.raises(RuntimeError, match="Syn_Seq aggregator not yet fitted"):
        syn_seq.generate_col(3, label_encoder={})

