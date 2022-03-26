# stdlib
import logging
import os
import sys
import warnings

# synthcity relative
from . import logger  # noqa: F401

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

warnings.simplefilter(action="ignore")

logger.add(sink=sys.stderr, level="CRITICAL")
