# methods/__init__.py

# For each file, import the relevant "syn_*" function (and the "generate_*" if you want it accessible).
# synthcity relative
from .cart import generate_cart, syn_cart
from .ctree import generate_ctree, syn_ctree
from .logreg import generate_logreg, syn_logreg
from .misc import (
    generate_lognorm,
    generate_random,
    generate_swr,
    syn_lognorm,
    syn_random,
    syn_swr,
)
from .norm import generate_norm, syn_norm
from .pmm import generate_pmm, syn_pmm
from .polyreg import generate_polyreg, syn_polyreg
from .rf import generate_rf, syn_rf

__all__ = [
    "syn_cart",
    "generate_cart",
    "syn_ctree",
    "generate_ctree",
    "syn_logreg",
    "generate_logreg",
    "syn_norm",
    "generate_norm",
    "syn_pmm",
    "generate_pmm",
    "syn_polyreg",
    "generate_polyreg",
    "syn_rf",
    "generate_rf",
    "syn_lognorm",
    "generate_lognorm",
    "syn_random",
    "generate_random",
    "syn_swr",
    "generate_swr",
]
