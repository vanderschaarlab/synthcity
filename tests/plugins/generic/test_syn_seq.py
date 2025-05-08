# stdlib
from typing import Any

# third party
import numpy as np
import pandas as pd
import pytest

# synthcity absolute
from synthcity.plugins.core.dataloader import Syn_SeqDataLoader
from synthcity.plugins.core.models.syn_seq.syn_seq import METHOD_MAP
from synthcity.plugins.core.schema import Schema

# Import the plugin and core components.
from synthcity.plugins.generic.plugin_syn_seq import Syn_SeqPlugin


# --- Dummy synthesis functions ---
def dummy_syn_fit(y: Any, X: Any, random_state: int = 0) -> str:
    """A dummy fit function that returns a marker string."""
    return "dummy_model"


def dummy_generate(dummy_model: Any, Xsyn: Any) -> Any:
    """A dummy generate function that adds 100 to every numeric value in the input."""
    # Assuming Xsyn is a numeric array
    return Xsyn.flatten() + 100


# Patch the METHOD_MAP so that the "cart" method uses our dummy functions.
METHOD_MAP["cart"] = (dummy_syn_fit, dummy_generate)


# --- Pytest Fixtures ---


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Returns a simple DataFrame with two numeric columns: A and B."""
    return pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})


@pytest.fixture
def plugin_instance() -> Syn_SeqPlugin:
    """Instantiates a Syn_SeqPlugin with fixed random_state and sampling_patience."""
    return Syn_SeqPlugin(random_state=42, sampling_patience=100)


@pytest.fixture
def schema_from_df(sample_df: pd.DataFrame) -> Schema:
    """Creates a Schema based on the sample DataFrame."""
    return Schema(data=sample_df)


# --- Test Function ---
def test_syn_seq_plugin(
    sample_df: pd.DataFrame, plugin_instance: Syn_SeqPlugin
) -> None:
    """
    End-to-end test of the Syn_SeqPlugin.
      1. Fit the plugin on a simple DataFrame.
      2. Manually override internal Syn_Seq attributes to simulate a fitted model.
      3. Generate synthetic data and verify that for each generated row, the value in B equals A + 100.
    """
    # Disable strict mode to bypass the synthetic constraints check.
    plugin_instance.strict = False

    # Fit the plugin on the DataFrame.
    plugin_instance.fit(sample_df)

    # Ensure the model is not None before proceeding.
    model = plugin_instance.model  # This is an instance of Syn_Seq.
    assert model is not None, "Model should not be None after fitting."

    # For testing, override internal model attributes of the underlying Syn_Seq instance.
    model._syn_order = ["A", "B"]
    model._method_map = {"B": "cart"}
    model._varsel = {"B": ["A"]}
    model._first_col_values = {"A": np.array(sample_df["A"])}
    model._model_trained = True
    model.cat_distributions = (
        {}
    )  # (For this dummy test, we leave the categorical distributions empty)
    model._col_models = {"B": {"name": "cart", "fitted_model": "dummy_model"}}

    # Also simulate plugin-level attributes (data encoders and data info).
    plugin_instance._data_encoders = {}  # No encoders needed for this dummy test.
    plugin_instance.data_info = {
        "data_type": "syn_seq",
        "syn_order": ["A", "B"],
        "method": {"B": "cart"},
        "variable_selection": {"B": ["A"]},
        "user_custom": {},
    }

    # Generate synthetic data (we request 3 rows).
    generated = plugin_instance.generate(count=3)

    # Check that the generated object is a Syn_SeqDataLoader.
    assert isinstance(generated, Syn_SeqDataLoader)
    gen_df = generated.dataframe()

    # Verify that the DataFrame has exactly the two expected columns and three rows.
    assert list(gen_df.columns) == ["A", "B"]
    assert len(gen_df) == 3

    # For each generated row, column B should equal column A + 100.
    for i in range(3):
        a_val = gen_df.loc[i, "A"]
        b_val = gen_df.loc[i, "B"]
        np.testing.assert_almost_equal(b_val, a_val + 100)
