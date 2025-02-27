import numpy as np
import pytest

# --- Import the synthesis methods ---
from synthcity.plugins.core.models.syn_seq.methods import (
    syn_cart, generate_cart,
    syn_ctree, generate_ctree,
    syn_logreg, generate_logreg,
    syn_norm, generate_norm,
    syn_pmm, generate_pmm,
    syn_polyreg, generate_polyreg,
    syn_rf, generate_rf,
    syn_lognorm, generate_lognorm,
    syn_random, generate_random,
    syn_swr, generate_swr,
)

# Map each method name to its corresponding (synthesis, generation) function pair.
METHOD_PAIRS = {
    "cart": (syn_cart, generate_cart),
    "ctree": (syn_ctree, generate_ctree),
    "logreg": (syn_logreg, generate_logreg),
    "norm": (syn_norm, generate_norm),
    "pmm": (syn_pmm, generate_pmm),
    "polyreg": (syn_polyreg, generate_polyreg),
    "rf": (syn_rf, generate_rf),
    "lognorm": (syn_lognorm, generate_lognorm),
    "random": (syn_random, generate_random),
    "swr": (syn_swr, generate_swr),
}

@pytest.mark.parametrize("method_name", list(METHOD_PAIRS.keys()))
def test_methods_output_shape(method_name: str):
    """
    Test that each synthesis method:
      - Returns a fitted model (non-None),
      - Generates a 1D numpy array with the same number of elements as the target vector.
    """
    syn_func, gen_func = METHOD_PAIRS[method_name]
    # Create dummy target data (10 values) and predictor data (10 rows, 3 columns)
    y = np.linspace(1, 10, num=10)
    X = np.tile(np.arange(3), (10, 1)).astype(float)
    
    # Fit the synthesis model.
    fitted_model = syn_func(y, X, random_state=42)
    assert fitted_model is not None, f"Method '{method_name}': fitted model is None"
    
    # Generate synthetic output.
    y_syn = gen_func(fitted_model, X)
    
    # Verify that the output is a numpy array, is 1D, and has the correct length.
    assert isinstance(y_syn, np.ndarray), f"Method '{method_name}': output is not a numpy array"
    assert y_syn.ndim == 1, f"Method '{method_name}': output is not 1D"
    np.testing.assert_equal(
        y_syn.shape[0], y.shape[0],
        err_msg=f"Method '{method_name}': output length {y_syn.shape[0]} does not match target length {y.shape[0]}"
    )

@pytest.mark.parametrize("method_name", list(METHOD_PAIRS.keys()))
def test_methods_numeric_output(method_name: str):
    """
    Test that each synthesis method generates output with a numeric dtype.
    """
    syn_func, gen_func = METHOD_PAIRS[method_name]
    y = np.linspace(0, 1, num=10)
    X = np.random.rand(10, 5)
    
    fitted_model = syn_func(y, X, random_state=0)
    y_syn = gen_func(fitted_model, X)
    
    # Verify that the output array has a numeric dtype.
    assert np.issubdtype(y_syn.dtype, np.number), (
        f"Method '{method_name}': output dtype {y_syn.dtype} is not numeric"
    )

def test_methods_invalid_input():
    """
    Test that providing mismatched dimensions between y and X raises an exception.
    (Assuming that the synthesis functions expect y and X to have the same number of rows.)
    """
    for method_name, (syn_func, _) in METHOD_PAIRS.items():
        y = np.arange(5, dtype=float)             # 5 elements in target vector.
        X = np.arange(12, dtype=float).reshape(4, 3)  # 4 rows in predictor matrix (mismatch).
        with pytest.raises(Exception):
            syn_func(y, X, random_state=0)
