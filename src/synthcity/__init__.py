# stdlib
import logging
import os
import sys
import warnings

# third party
import optuna
import pandas as pd

# synthcity relative
from . import logger  # noqa: F401

optuna.logging.set_verbosity(optuna.logging.FATAL)
optuna.logging.disable_propagation()
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

pd.options.mode.chained_assignment = None

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

warnings.simplefilter(action="ignore")

logger.add(sink=sys.stderr, level="ERROR")
