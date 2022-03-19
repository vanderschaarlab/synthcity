# stdlib
import sys
import warnings

# synthcity relative
from . import logger  # noqa: F401

warnings.simplefilter(action="ignore", category=FutureWarning)
logger.add(sink=sys.stderr, level="CRITICAL")
