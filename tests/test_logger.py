# stdlib
import sys
from typing import Callable

# third party
import pytest

# synthcity absolute
import synthcity.logger as log


@pytest.mark.parametrize("level", ["ERROR", "DEBUG", "CRITICAL"])
def test_loglevel(level: str) -> None:
    log.add(sink=sys.stderr, level=level)
    log.remove()


@pytest.mark.parametrize(
    ("level", "cbk"),
    [
        ("ERROR", log.error),
        ("DEBUG", log.debug),
        ("WARNING", log.warning),
        ("CRITICAL", log.critical),
    ],
)
def test_log_cbk(level: str, cbk: Callable) -> None:
    log.add(sink=sys.stderr, level=level)
    cbk("test")
    log.remove()
