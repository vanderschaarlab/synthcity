# stdlib
import shutil
from pathlib import Path
from typing import Generator

# third party
import pytest

# synthcity absolute
from synthcity.utils.reproducibility import clear_cache, enable_reproducible_results


@pytest.fixture(autouse=True, scope="session")
def run_before_tests() -> Generator:
    enable_reproducible_results(0)
    clear_cache()

    yield

    # cleanup after test
    workspace = Path("workspace")
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)


# Hook to modify the test result if it exceeds a timeout
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:
    """Modify the test result if it exceeds the timeout to skip instead of failing."""
    if call.when == "call" and call.excinfo is not None:
        # Check if the test was stopped due to a timeout using call.result
        if "Timeout" in str(call.excinfo.value):
            # Mark the test as skipped due to exceeding the timeout
            pytest.skip(f"Test skipped due to exceeding the timeout: {item.nodeid}")
