# stdlib
import sys

# synthcity relative
from . import logger  # noqa: F401

logger.add(sink=sys.stderr, level="CRITICAL")
