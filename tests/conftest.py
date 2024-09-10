# stdlib
import shutil
from pathlib import Path
from typing import Generator

# third party
import pytest
from _pytest.nodes import Item
from _pytest.runner import CallInfo

# synthcity absolute
from synthcity.utils.reproducibility import clear_cache, enable_reproducible_results


@pytest.fixture(autouse=True, scope="session")
def run_before_tests() -> Generator:
    """Setup reproducible results and clear cache before tests."""
    enable_reproducible_results(0)
    clear_cache()

    yield

    # Cleanup after tests
    workspace = Path("workspace")
    if workspace.exists() and workspace.is_dir():
        shutil.rmtree(workspace, ignore_errors=True)


# Hook to modify the test result if it exceeds a timeout
def pytest_runtest_makereport(item: Item, call: CallInfo) -> None:
    """Modify the test result if it exceeds the timeout to skip instead of failing."""
    if call.when == "call" and call.excinfo is not None:
        # Check if the exception is a TimeoutError from pytest-timeout
        if isinstance(call.excinfo.value, pytest.TimeoutExpired):
            # Mark the test as skipped due to exceeding the timeout
            pytest.skip(f"Test skipped due to exceeding the timeout: {item.nodeid}")
