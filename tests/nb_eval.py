# stdlib
from pathlib import Path
from time import time
from typing import List

# third party
import click
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

workspace = Path(__file__).parents[0] / "workspace"
workspace.mkdir(parents=True, exist_ok=True)


def run_notebook(notebook_path: Path) -> None:
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    proc = ExecutePreprocessor(timeout=1800)
    # Will raise on cell error
    proc.preprocess(nb, {"metadata": {"path": workspace}})


try:
    # synthcity absolute
    from synthcity.plugins.core.models.tabular_goggle import TabularGoggle  # noqa: F401

    goggle_disabled = False
except ImportError:
    goggle_disabled = True

try:
    # synthcity absolute
    from synthcity.plugins.core.models.tabular_arf import TabularARF  # noqa: F401

    arf_disabled = False
except ImportError:
    arf_disabled = True

try:
    # synthcity absolute
    from synthcity.plugins.core.models.tabular_great import TabularGReaT  # noqa: F401

    great_disabled = False
except ImportError:
    great_disabled = True

all_tests = [
    "basic_examples",
    "benchmarks",
    "survival_analysis",
    "time_series",
    "time_series_data_preparation",
    "hyperparameter_optimization",
    "plugin_adsgan",
    "plugin_ctgan",
    "plugin_nflow",
    "plugin_tvae",
    "plugin_timegan",
    "plugin_radialgan" "plugin_arf",
    "plugin_bayesian_network",
    "plugin_ddpm",
    "plugin_dummy_sampler",
    "plugin_marginal_distribution",
    "plugin_uniform_sampler",
    "plugin_image_adsgan",
    "plugin_image_cgan",
    "plugin_decaf",
    "plugin_dpgan",
    "plugin_pategan",
    "plugin_privbayes",
    "plugin_ctgan(generic)",
    "plugin_fourier_flows",
    "plugin_timegan",
    "plugin_aim",
]

if not goggle_disabled:
    all_tests.append("plugin_goggle")
if not arf_disabled:
    all_tests.append("plugin_arf")
if not great_disabled:
    all_tests.append("plugin_great")

minimal_tests = [
    "basic_examples",
    "plugin_adsgan",
    "plugin_ctgan",
    "plugin_nflow",
    "plugin_tvae",
    "plugin_timegan",
]

# For extras
goggle_tests = ["plugin_goggle"]
arf_tests = ["plugin_arf"]
great_tests = ["plugin_great"]


@click.command()
@click.option("--nb_dir", type=str, default=".")
@click.option(
    "--tutorial_tests",
    type=click.Choice(
        ["minimal_tests", "all_tests", "goggle_tests", "plugin_arf", "plugin_great"],
        case_sensitive=False,
    ),
    default="minimal_tests",
)
def main(nb_dir: Path, tutorial_tests: str) -> None:
    nb_dir = Path(nb_dir)
    enabled_tests: List = []
    if tutorial_tests == "all_tests":
        enabled_tests = all_tests
    elif tutorial_tests == "minimal_tests":
        enabled_tests = minimal_tests

    for p in nb_dir.rglob("*"):
        if p.suffix != ".ipynb":
            continue

        if "checkpoint" in p.name:
            continue

        ignore = True
        for val in enabled_tests:
            if val in p.name:
                ignore = False
                break

        if ignore:
            continue

        print("Testing ", p.name)
        start = time()
        try:
            run_notebook(p)
        except BaseException as e:
            print("FAIL", p.name, e)

            raise e
        finally:
            print(f"Tutorial {p.name} tool {time() - start}")


if __name__ == "__main__":
    main()
