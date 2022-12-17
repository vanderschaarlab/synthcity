# stdlib
import shutil
from pathlib import Path

# third party
import pytest

# synthcity absolute
from synthcity.utils.reproducibility import clear_cache, enable_reproducible_results


@pytest.fixture(autouse=True)
def run_before_tests() -> None:
    enable_reproducible_results(0)
    clear_cache()

    workspace = Path("workspace")
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
