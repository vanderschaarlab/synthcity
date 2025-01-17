# methods/__init__.py

# For each file, import the relevant "syn_*" function (and the "generate_*" if you want it accessible).
from .cart import syn_cart, generate_cart
from .ctree import syn_ctree, generate_ctree
from .logreg import syn_logreg, generate_logreg
from .norm import syn_norm, generate_norm
from .pmm import syn_pmm, generate_pmm
from .polyreg import syn_polyreg, generate_polyreg
from .rf import syn_rf, generate_rf
from .misc import syn_lognorm, generate_lognorm
from .misc import syn_random, generate_random
from .misc import syn_swr, generate_swr

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
