import numpy as np
import pytest

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
    y = np.linspace(1, 10, num=10)
    X = np.tile(np.arange(3), (10, 1)).astype(float)
    
    fitted_model = syn_func(y, X, random_state=42)
    assert fitted_model is not None, f"Method '{method_name}': fitted model is None"
    
    y_syn = gen_func(fitted_model, X)
    
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
    
    assert np.issubdtype(y_syn.dtype, np.number), (
        f"Method '{method_name}': output dtype {y_syn.dtype} is not numeric"
    )

