import numpy as np
import pandas as pd
import pytest

# Import the plugin and internal components.
from synthcity.plugins.generic.plugin_syn_seq import Syn_SeqPlugin
from synthcity.plugins.core.models.syn_seq.syn_seq import Syn_Seq, METHOD_MAP
from synthcity.plugins.core.dataloader import Syn_SeqDataLoader
from synthcity.plugins.core.schema import Schema

# --- Dummy functions for synthesis ---
def dummy_syn_fit(y, X, random_state=0):
    """Dummy fit function that simply returns a marker."""
    return "dummy_model"

def dummy_generate(dummy_model, Xsyn):
    """Dummy generate function that adds 100 to each numeric value."""
    return Xsyn.flatten() + 100

# Patch the METHOD_MAP so that the "cart" method uses our dummy functions.
METHOD_MAP["cart"] = (dummy_syn_fit, dummy_generate)

# --- Fixtures ---
@pytest.fixture
def sample_df():
    """A simple DataFrame with two numeric columns A and B."""
    return pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50]
    })

@pytest.fixture
def plugin_instance():
    """Instantiate the Syn_SeqPlugin with fixed random_state and patience."""
    return Syn_SeqPlugin(random_state=42, sampling_patience=100)

@pytest.fixture
def schema_from_df(sample_df):
    """Create a Schema based on the sample DataFrame."""
    return Schema(data=sample_df)

# --- Test Function ---
def test_plugin_syn_seq(sample_df, plugin_instance, schema_from_df):
    """
    Test the Syn_SeqPlugin end-to-end:
      - Fit the plugin on a simple DataFrame.
      - Override internal Syn_Seq attributes to simulate a fitted model.
      - Generate synthetic data and verify that for each row, column B equals column A + 100.
    """
    # Fit the plugin on the DataFrame.
    plugin_instance.fit(sample_df)
    
    # For testing purposes, override internal attributes of the underlying Syn_Seq model.
    model = plugin_instance.model  # This should be an instance of Syn_Seq.
    model._syn_order = ["A", "B"]
    model._method_map = {"B": "cart"}
    model._varsel = {"B": ["A"]}
    model._first_col_values = {"A": np.array(sample_df["A"])}
    model._model_trained = True
    model.cat_distributions = {}  # Empty dictionary (if needed)
    model._col_models = {"B": {"name": "cart", "fitted_model": "dummy_model"}}
    
    # Also, simulate plugin-level attributes.
    plugin_instance._data_encoders = {}  # Dummy data encoders.
    plugin_instance.data_info = {
        "syn_order": ["A", "B"],
        "method": {"B": "cart"},
        "variable_selection": {"B": ["A"]},
        "user_custom": {}
    }
    
    # Generate synthetic data (count = 3) using the plugin.
    generated = plugin_instance.generate(count=3, syn_schema=schema_from_df)
    
    # Verify that the generated object is a Syn_SeqDataLoader.
    assert isinstance(generated, Syn_SeqDataLoader)
    gen_df = generated.dataframe()
    
    # Check that the DataFrame has exactly the two expected columns and three rows.
    assert list(gen_df.columns) == ["A", "B"]
    assert len(gen_df) == 3
    
    # Verify that for each row, the synthetic value for B equals (A + 100).
    for i in range(3):
        a_val = gen_df.loc[i, "A"]
        b_val = gen_df.loc[i, "B"]
        np.testing.assert_almost_equal(b_val, a_val + 100)
